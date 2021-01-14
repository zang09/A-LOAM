// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <vector>
#include <string>
#include <mutex>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using namespace std;
using std::atan2;
using std::cos;
using std::sin;

struct PointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,
                                   (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring) (float, time, time)
                                   )

class ScanRegistration
{
private:
  ros::NodeHandle nh;

  ros::Publisher pubLaserCloud;
  ros::Publisher pubCornerPointsSharp;
  ros::Publisher pubCornerPointsLessSharp;
  ros::Publisher pubSurfPointsFlat;
  ros::Publisher pubSurfPointsLessFlat;
  ros::Publisher pubRemovePoints;
  std::vector<ros::Publisher> pubEachScan;

  ros::Subscriber subLaserCloud;
  ros::Subscriber subImuData;

  std::string pointCloudTopic;
  std::string imuTopic;

  pcl::PointCloud<PointXYZIRT>::Ptr   laserCloudInTime;
  pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn;

  int cloudSortInd[400000];
  int cloudNeighborPicked[400000];
  int cloudLabel[400000];

  double scanPeriod;
  int systemDelay;
  int systemInitCount;
  bool systemInited;
  int N_SCANS;
  bool PUB_EACH_LINE;
  double MINIMUM_RANGE;

  double timeScanCur;
  double timeScanEnd;
  std_msgs::Header cloudHeader;

  //IMU
  bool useImu;
  int imuPointerCur;
  bool firstPointFlag;
  int deskewFlag;
  bool imuAvailable;

  Eigen::Affine3f transStartInverse;
  std::deque<sensor_msgs::Imu> imuQueue;
  std::mutex imuLock;
  vector<double> extRotV;
  vector<double> extRPYV;
  vector<double> extTransV;
  Eigen::Matrix3d extRot;
  Eigen::Matrix3d extRPY;
  Eigen::Vector3d extTrans;
  Eigen::Quaterniond extQRPY;

  double *imuTime = new double[2000];
  double *imuRotX = new double[2000];
  double *imuRotY = new double[2000];
  double *imuRotZ = new double[2000];

public:
  static float cloudCurvature[400000];
  static bool compare (int i, int j)
  {
    return (cloudCurvature[i] < cloudCurvature[j]);
  }

public:
  ScanRegistration() :
    scanPeriod(0.1),
    systemDelay(0),
    systemInitCount(0),
    systemInited(false),
    N_SCANS(0),
    PUB_EACH_LINE(false),
    MINIMUM_RANGE(0.1),
    firstPointFlag(false),
    deskewFlag(0),
    imuAvailable(false)
  {
    initParams();
    subscribeAndPublisher();
  }

  ~ScanRegistration()
  {
  }

  void initParams()
  {
    //Get param
    nh.param<bool>("aloam_velodyne/use_imu", useImu, false);
    nh.param<std::string>("aloam_velodyne/pointcloud_topic", pointCloudTopic, "velodyne_points");
    nh.param<std::string>("aloam_velodyne/imu_topic", imuTopic, "imu/data");

    nh.param<int>("aloam_velodyne/scan_line", N_SCANS, 16);
    nh.param<double>("aloam_velodyne/minimum_range", MINIMUM_RANGE, 0.1);
    nh.param<vector<double>>("aloam_velodyne/extrinsic_rot", extRotV, vector<double>());
    nh.param<vector<double>>("aloam_velodyne/extrinsic_rpy", extRPYV, vector<double>());
    nh.param<vector<double>>("aloam_velodyne/extrinsic_trans", extTransV, vector<double>());

    extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
    extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
    extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
    extQRPY = Eigen::Quaterniond(extRPY);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
      printf("only support velodyne with 16, 32 or 64 scan line!");
      return;
    }

    laserCloudInTime.reset(new pcl::PointCloud<PointXYZIRT>());
    laserCloudIn.reset(new pcl::PointCloud<pcl::PointXYZ>());
  }

  void resetParams()
  {
    firstPointFlag = true;
    imuPointerCur = 0;

    for (int i = 0; i < 2000; ++i)
    {
        imuTime[i] = 0;
        imuRotX[i] = 0;
        imuRotY[i] = 0;
        imuRotZ[i] = 0;
    }

    laserCloudInTime->clear();
    laserCloudIn->clear();
  }

  void subscribeAndPublisher()
  {
    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 100, &ScanRegistration::laserCloudHandler, this,ros::TransportHints().tcpNoDelay());
    subImuData    = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ScanRegistration::imuDataHandler, this, ros::TransportHints().tcpNoDelay());

    pubLaserCloud            = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
    pubCornerPointsSharp     = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    pubSurfPointsFlat        = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    pubSurfPointsLessFlat    = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    pubRemovePoints          = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
      for(int i = 0; i < N_SCANS; i++)
      {
        ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
        pubEachScan.push_back(tmp);
      }
    }
  }

  void imuDataHandler(const sensor_msgs::ImuConstPtr &imuDataMsg)
  {
    sensor_msgs::Imu thisImu = imuConverter(*imuDataMsg);

    std::lock_guard<std::mutex> lock(imuLock);
    imuQueue.push_back(thisImu);
  }

  bool deskewImu()
  {
    std::lock_guard<std::mutex> lock(imuLock);

    // make sure IMU data available for the scan
    if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
    {
      ROS_DEBUG("Waiting for IMU data ...");
      return false;
    }

    imuAvailable = false;

    while (!imuQueue.empty())
    {
      if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
        imuQueue.pop_front();
      else
        break;
    }

    if (imuQueue.empty())
      return false;

    imuPointerCur = 0;

    for (int i = 0; i < (int)imuQueue.size(); ++i)
    {
      sensor_msgs::Imu thisImuMsg = imuQueue[i];
      double currentImuTime = thisImuMsg.header.stamp.toSec();

      // Get roll, pitch, and yaw estimation for this scan
      if (currentImuTime <= timeScanCur)
      {
        double imuRoll, imuPitch, imuYaw; // not used now
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(thisImuMsg.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
      }

      if (currentImuTime > timeScanEnd + 0.01)
        break;

      if (imuPointerCur == 0)
      {
        imuRotX[0] = 0;
        imuRotY[0] = 0;
        imuRotZ[0] = 0;
        imuTime[0] = currentImuTime;
        ++imuPointerCur;
        continue;
      }

      // Get angular velocity
      double angular_x, angular_y, angular_z;
      angular_x = thisImuMsg.angular_velocity.x;
      angular_y = thisImuMsg.angular_velocity.y;
      angular_z = thisImuMsg.angular_velocity.z;

      // integrate rotation
      double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
      imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
      imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
      imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
      imuTime[imuPointerCur] = currentImuTime;
      ++imuPointerCur;
    }

    --imuPointerCur;

    if (imuPointerCur <= 0)
      return false;

    imuAvailable = true;

    return true;
  }

  void projectPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
  {
    int cloudSize = laserCloudInTime->points.size();

    // range image projection
    for (int i = 0; i < cloudSize; ++i)
    {
        pcl::PointXYZ thisPoint;
        thisPoint.x = laserCloudInTime->points[i].x;
        thisPoint.y = laserCloudInTime->points[i].y;
        thisPoint.z = laserCloudInTime->points[i].z;
        //thisPoint.intensity = laserCloudInTime->points[i].intensity;

        float range = pointDistance(thisPoint);
        if (range < 1.0 || range > 1000.0)
            continue;

        int rowIdn = laserCloudInTime->points[i].ring;
        if (rowIdn < 0 || rowIdn >= N_SCANS)
            continue;

        //if (rowIdn % downsampleRate != 0)
        //    continue;

        float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        static float ang_res_x = 360.0/float(1800);
        int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + 1800/2;
        if (columnIdn >= 1800)
            columnIdn -= 1800;

        if (columnIdn < 0 || columnIdn >= 1800)
            continue;

        thisPoint = deskewPoint(&thisPoint, laserCloudInTime->points[i].time); // Velodyne
        cloud->push_back(thisPoint);
    }
  }

  pcl::PointXYZ deskewPoint(pcl::PointXYZ *point, double relTime)
  {
    if (deskewFlag == -1 || imuAvailable == false)
        return *point;

    double pointTime = timeScanCur + relTime;

    float rotXCur, rotYCur, rotZCur;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur=0, posYCur=0, posZCur=0; // not used now

    if (firstPointFlag == true)
    {
        transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
        firstPointFlag = false;
    }

    // transform points to start
    Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBt = transStartInverse * transFinal;

    pcl::PointXYZ newPoint;
    newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
    newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
    newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
    //newPoint.intensity = point->intensity;

    return newPoint;
  }

  float pointDistance(pcl::PointXYZ p)
  {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
  }

  void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
  {
      *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

      int imuPointerFront = 0;
      while (imuPointerFront < imuPointerCur)
      {
          if (pointTime < imuTime[imuPointerFront])
              break;
          ++imuPointerFront;
      }

      if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
      {
          *rotXCur = imuRotX[imuPointerFront];
          *rotYCur = imuRotY[imuPointerFront];
          *rotZCur = imuRotZ[imuPointerFront];
      }
      else
      {
          int imuPointerBack = imuPointerFront - 1;
          double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
          double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
          *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
          *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
          *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
      }
  }

  void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
  {
    resetParams();

    if (!systemInited)
    {
      systemInitCount++;
      if (systemInitCount >= systemDelay)
      {
        systemInited = true;
      }
      else
        return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::fromROSMsg(*laserCloudMsg, *laserCloudInTime);

    /* new code here */
    if(useImu)
    {
      // Get timestamp
      cloudHeader = laserCloudMsg->header;
      timeScanCur = cloudHeader.stamp.toSec();
      timeScanEnd = timeScanCur + laserCloudInTime->points.back().time; // Velodyne

      // check point time
      if (deskewFlag == 0)
      {
          deskewFlag = -1;
          for (int i = 0; i < (int)laserCloudMsg->fields.size(); ++i)
          {
              if (laserCloudMsg->fields[i].name == "time")
              {
                  deskewFlag = 1;
                  break;
              }
          }
          if (deskewFlag == -1)
              ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
      }

      if(!deskewImu())
        return;

      projectPointCloud(laserCloudIn);
    }

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    removeClosedPointCloud(*laserCloudIn, *laserCloudIn, MINIMUM_RANGE);

    int cloudSize = laserCloudIn->points.size();
    float startOri = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
    float endOri = -atan2(laserCloudIn->points[cloudSize - 1].y, laserCloudIn->points[cloudSize - 1].x) + 2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
      endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
      endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
      point.x = laserCloudIn->points[i].x;
      point.y = laserCloudIn->points[i].y;
      point.z = laserCloudIn->points[i].z;

      float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
      int scanID = 0;

      if (N_SCANS == 16)
      {
        scanID = int((angle + 15) / 2 + 0.5);
        if (scanID > (N_SCANS - 1) || scanID < 0)
        {
          count--;
          continue;
        }
      }
      else if (N_SCANS == 32)
      {
        scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
        if (scanID > (N_SCANS - 1) || scanID < 0)
        {
          count--;
          continue;
        }
      }
      else if (N_SCANS == 64)
      {
        if (angle >= -8.83)
          scanID = int((2 - angle) * 3.0 + 0.5);
        else
          scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

        // use [0 50]  > 50 remove outlies
        if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
        {
          count--;
          continue;
        }
      }
      else
      {
        printf("wrong scan number\n");
        ROS_BREAK();
      }
      //printf("angle %f scanID %d \n", angle, scanID);

      float ori = -atan2(point.y, point.x);
      if (!halfPassed)
      {
        if (ori < startOri - M_PI / 2)
        {
          ori += 2 * M_PI;
        }
        else if (ori > startOri + M_PI * 3 / 2)
        {
          ori -= 2 * M_PI;
        }

        if (ori - startOri > M_PI)
        {
          halfPassed = true;
        }
      }
      else
      {
        ori += 2 * M_PI;
        if (ori < endOri - M_PI * 3 / 2)
        {
          ori += 2 * M_PI;
        }
        else if (ori > endOri + M_PI / 2)
        {
          ori -= 2 * M_PI;
        }
      }

      float relTime = (ori - startOri) / (endOri - startOri);
      point.intensity = scanID + scanPeriod * relTime;
      laserCloudScans[scanID].push_back(point);
    }

    cloudSize = count;
    //printf("points size %d \n", cloudSize);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    {
      scanStartInd[i] = laserCloud->size() + 5;
      *laserCloud += laserCloudScans[i];
      scanEndInd[i] = laserCloud->size() - 6;
    }

    //printf("prepare time %f \n", t_prepare.toc());

    for (int i = 5; i < cloudSize - 5; i++)
    {
      float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
      float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
      float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

      cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
      cloudSortInd[i] = i;
      cloudNeighborPicked[i] = 0;
      cloudLabel[i] = 0;
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
      if( scanEndInd[i] - scanStartInd[i] < 6)
        continue;
      pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
      for (int j = 0; j < 6; j++)
      {
        int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
        int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

        TicToc t_tmp;
        std::sort (cloudSortInd+sp, cloudSortInd+ep+1, compare);
        t_q_sort += t_tmp.toc();

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--)
        {
          int ind = cloudSortInd[k];

          if (cloudNeighborPicked[ind] == 0 &&
              cloudCurvature[ind] > 0.1)
          {

            largestPickedNum++;
            if (largestPickedNum <= 2)
            {
              cloudLabel[ind] = 2;
              cornerPointsSharp.push_back(laserCloud->points[ind]);
              cornerPointsLessSharp.push_back(laserCloud->points[ind]);
            }
            else if (largestPickedNum <= 20)
            {
              cloudLabel[ind] = 1;
              cornerPointsLessSharp.push_back(laserCloud->points[ind]);
            }
            else
            {
              break;
            }

            cloudNeighborPicked[ind] = 1;

            for (int l = 1; l <= 5; l++)
            {
              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
              if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
              {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--)
            {
              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
              if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
              {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        int smallestPickedNum = 0;
        for (int k = sp; k <= ep; k++)
        {
          int ind = cloudSortInd[k];

          if (cloudNeighborPicked[ind] == 0 &&
              cloudCurvature[ind] < 0.1)
          {

            cloudLabel[ind] = -1;
            surfPointsFlat.push_back(laserCloud->points[ind]);

            smallestPickedNum++;
            if (smallestPickedNum >= 4)
            {
              break;
            }

            cloudNeighborPicked[ind] = 1;
            for (int l = 1; l <= 5; l++)
            {
              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
              if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
              {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--)
            {
              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
              if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
              {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++)
        {
          if (cloudLabel[k] <= 0)
          {
            surfPointsLessFlatScan->push_back(laserCloud->points[k]);
          }
        }
      }

      pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
      pcl::VoxelGrid<PointType> downSizeFilter;
      downSizeFilter.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter.setLeafSize(0.2, 0.2, 0.2); // 0.2, 0.2, 0.2
      downSizeFilter.filter(surfPointsLessFlatScanDS);

      surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    //printf("sort q time %f \n", t_q_sort);
    //printf("seperate points time %f \n", t_pts.toc());


    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/aft_mapped";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/aft_mapped";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/aft_mapped";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/aft_mapped";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/aft_mapped";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scam
    if(PUB_EACH_LINE)
    {
      for(int i = 0; i< N_SCANS; i++)
      {
        sensor_msgs::PointCloud2 scanMsg;
        pcl::toROSMsg(laserCloudScans[i], scanMsg);
        scanMsg.header.stamp = laserCloudMsg->header.stamp;
        scanMsg.header.frame_id = "/aft_mapped";
        pubEachScan[i].publish(scanMsg);
      }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
      ROS_WARN("scan registration process over 100ms");
  }

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
  {
    sensor_msgs::Imu imu_out = imu_in;

    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);

    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();

    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);

    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();

    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
    {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }

    return imu_out;
  }

  template <typename PointT>
  void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
  {
    if (&cloud_in != &cloud_out)
    {
      cloud_out.header = cloud_in.header;
      cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
      if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
        continue;
      cloud_out.points[j] = cloud_in.points[i];
      j++;
    }
    if (j != cloud_in.points.size())
    {
      cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
  }
};

float ScanRegistration::cloudCurvature[];

int main(int argc, char **argv)
{
  ros::init(argc, argv, "scanRegistration");

  ScanRegistration SR;

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
  //ros::spin();

  return 0;
}
