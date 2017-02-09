// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {
using namespace cv;
    using namespace Eigen;
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  detectFeatures(frame_ref, px_ref_, f_ref_);
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;


  cv::Mat img_cur, img_ref;
  img_cur = frame_cur->img_pyr_[0];img_ref = frame_ref_->img_pyr_[0];
  cv::Mat img_show1( img_cur.rows, img_cur.cols*2, CV_8UC1 );
  img_ref.copyTo( img_show1( cv::Rect(0,0,img_ref.cols, img_ref.rows) ) );
  img_cur.copyTo( img_show1(cv::Rect(img_cur.cols,0,img_cur.cols, img_cur.rows)) );
  for (int i = 0; i < px_cur_.size(); i++)
  {
    float b = 255*float(rand())/RAND_MAX;
    float g = 255*float(rand())/RAND_MAX;
    float r = 255*float(rand())/RAND_MAX;
    cv::circle(img_show1, px_ref_[i], 5, cv::Scalar(b,g,r),1 );
    cv::circle(img_show1, cv::Point2d(px_cur_[i].x+img_ref.cols, px_cur_[i].y), 5, cv::Scalar(b,g,r),1 );
    cv::line( img_show1, px_ref_[i], cv::Point2d(px_cur_[i].x+img_ref.cols, px_cur_[i].y), cv::Scalar(b,g,r), 1 );
  }
  cv::imshow( "result", img_show1 );
  cv::waitKey(0);



    cv::Mat E, R, t, mark;
    E = cv::findEssentialMat(px_ref_, px_cur_, 329.11, cv::Point2d(320.0, 240.0), cv::RANSAC, 0.999, 1.0, mark);
    cv::recoverPose(E, px_ref_, px_cur_, R, t, 329.11, cv::Point2d(320.0, 240.0), mark);
    cout<<"essential_matrix is "<<endl<< E<<endl;

    Matrix3d ro ;
    ro<<    R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    Vector3d tt;
    tt<<t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0);
    T_cur_from_ref_ = Sophus::SE3(
            ro,
            tt);

  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);
  double scale = Config::mapScale()/scene_depth_median;
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
    vector<cv::Point2d> p_cur, p_ref;

  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
      p_cur.push_back(cv::Point2d(px_cur_[*it].x, px_cur_[*it].y));
      p_ref.push_back(cv::Point2d(px_ref_[*it].x, px_ref_[*it].y));
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }

    img_cur = frame_cur->img_pyr_[0];img_ref = frame_ref_->img_pyr_[0];
    //img_cur = imread("/home/chen/data/rpg_urban_pinhole_data/data/img/img0001_0.png",1);
    //img_ref = imread("/home/chen/data/rpg_urban_pinhole_data/data/img/img0041_0.png",1);
    cv::Mat img_show2( img_cur.rows, img_cur.cols*2, CV_8UC1 );
    img_ref.copyTo( img_show2( cv::Rect(0,0,img_ref.cols, img_ref.rows) ) );
    img_cur.copyTo( img_show2(cv::Rect(img_cur.cols,0,img_cur.cols, img_cur.rows)) );
    for (int i = 0; i < p_cur.size(); i++)
    {
        float b = 255*float(rand())/RAND_MAX;
        float g = 255*float(rand())/RAND_MAX;
        float r = 255*float(rand())/RAND_MAX;
        cv::circle(img_show2, p_ref[i], 5, cv::Scalar(b,g,r),1 );
        cv::circle(img_show2, cv::Point2d(p_cur[i].x+img_ref.cols, p_cur[i].y), 5, cv::Scalar(b,g,r),1 );
        cv::line( img_show2, p_ref[i], cv::Point2d(p_cur[i].x+img_ref.cols, p_cur[i].y), cv::Scalar(b,g,r), 1 );
    }
    cv::imshow( "result", img_show2 );
    cv::waitKey(0);

  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}

void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 50.0;
  const int klt_max_iter = 50;
  const double klt_eps = 0.0001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3 T_cur_from_ref)
{
  vector<Vector2d, aligned_allocator<Vector2d> > uv_ref(f_ref.size());
  vector<Vector2d, aligned_allocator<Vector2d> > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }

    vector<int> outliers;

    vector<cv::Point2f> ref_pts(uv_cur.size()), cur_pts(uv_cur.size());
    for(size_t i=0; i<uv_cur.size(); ++i)
    {
        ref_pts[i] = cv::Point2f(uv_ref[i][0], uv_ref[i][1]);
        cur_pts[i] = cv::Point2f(uv_cur[i][0], uv_cur[i][1]);
    }

    Matrix3d R;
    R = T_cur_from_ref.rotation_matrix();
    Vector3d t;
    t = T_cur_from_ref.translation();
    Mat T1 = (Mat_<float> (3,4) <<
            1,0,0,0,
            0,1,0,0,
            0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
             R(0,0), R(0,1), R(0,2), t(0,0),
            R(1,0), R(1,1), R(1,2), t(1,0),
            R(2,0), R(2,1), R(2,2), t(2,0)
    );

    Mat K = ( Mat_<double> ( 3,3 ) << 329.11, 0, 320.0, 0, 329.11, 240.0, 0, 0, 1 );

    Mat pts_4d;
    cv::triangulatePoints( T1, T2, ref_pts, cur_pts, pts_4d );

    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Vector3d p;
        p<<
            x.at<float>(0,0),
            x.at<float>(1,0),
            x.at<float>(2,0);
        xyz_in_cur.push_back( p );
        double e1 = vk::reprojError(f_cur[i], R*(xyz_in_cur.back()+t), focal_length);
        double e2 = vk::reprojError(f_ref[i], xyz_in_cur.back(), focal_length);
        //double e1(1.0),e2(1.0);
        cout<<"e1 = "<<e1<<"  e2 = "<<e2<<endl;
        if(e1 > reprojection_threshold || e2 > reprojection_threshold)
            outliers.push_back(i);
        else
        {
            inliers.push_back(i);
        }
    }
  //vk::computeInliers(f_cur, f_ref,
   //                  Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
   //                  reprojection_threshold, focal_length,
   //                  xyz_in_cur, inliers, outliers);
}


} // namespace initialization
} // namespace svo
