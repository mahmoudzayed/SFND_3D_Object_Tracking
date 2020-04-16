#include <numeric>
#include "matching2D.hpp"

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
  // configure matcher
  bool crossCheck = false;
//   std::cout<<"src_type : "<<descSource.type()<< ", dst_type : "<<descRef.type()<<std::endl;
//   std::cout<<"src_size " <<descSource.size() <<", dst_size: "<<descRef.size()<<std::endl;
 
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0)
  {
    int normType;
    if (descSource.type() != CV_8U || descRef.type() != CV_8U)
    { 
      normType = cv::NORM_L2;
    }
    else
    {
      normType = cv::NORM_HAMMING;
    }
    matcher = cv::BFMatcher::create(normType, crossCheck);
  }
  else if (matcherType.compare("MAT_FLANN") == 0)
  {
    if (descSource.type() != CV_32F || descRef.type() != CV_32F)
    { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
//     std::cout << "FLANN matching: ";
  }
//   std::cout<<" dtype : "<<descSource.type()<<std::endl;
//   std::cout<<" dtype : "<<descRef.type()<<std::endl;

  // perform matching task
  if (selectorType.compare("SEL_NN") == 0)
  { // nearest neighbor (best match)
    double t = (double)cv::getTickCount();
    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//     std::cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
  }
  else if (selectorType.compare("SEL_KNN") == 0)
  {  // k nearest neighbors (k=2)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    int k = 2;
    double t = (double)cv::getTickCount();
    // TODO : implement k-nearest-neighbor matching
    matcher->knnMatch(descSource, descRef, knn_matches, k);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//     std::cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms";
    // TODO : filter matches using descriptor distance ratio test
    double minDescDistRatio = 0.8;
    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
      if((*it)[0].distance < (minDescDistRatio*(*it)[1].distance))
      {
        matches.push_back((*it)[0]);
      }
    }
//     std::cout << "# keypoints after removal = " << matches.size() << std::endl;
//     std::cout<<","<<1000 * t / 1.0 <<","<<matches.size();
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0)//BRISK, BRIEF, ORB, FREAK, AKAZE and SIFT.
  {

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else
  {
    if (descriptorType.compare("BRIEF") == 0)
    {
      int bytes = 32;
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes);
    }
    if (descriptorType.compare("ORB") == 0)
    {
      int orbnfeatures = 500;
      float scaleFactor = 1.2f;
      int nlevels = 8;
      int edgeThreshold = 31;
      int firstLevel = 0;
      int WTA_K = 2;
      int scoreType = (int)cv::ORB::HARRIS_SCORE;
      int patchSize = 31;      
      extractor = cv::ORB::create(orbnfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K);
    }
    if (descriptorType.compare("FREAK") == 0)
    {
      bool orientationNormalized=true;
      bool scaleNormalized=true;
      float patternScale=22.0f;
      int nOctaves=4;
      std::vector<int> selectedPairs;
      extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    if (descriptorType.compare("AKAZE") == 0)
    {
      cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
      int descriptor_size=0;
      int descriptor_channels=3;
      float Akazethreshold=0.001f;
      int nOctaves=4;
      int nOctaveLayers=4;
      int diffusivity = cv::KAZE::DIFF_PM_G2;
      extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, Akazethreshold, nOctaves, nOctaveLayers);
      
    }
    if (descriptorType.compare("SIFT") == 0)
    {
      int siftnfeatures = 0;
      int nOctaveLayers = 3;
      double contrastThreshold = 0.04;
      double edgeThreshold = 10;
      double sigma = 1.6;
      extractor = cv::xfeatures2d::SIFT::create(siftnfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//   std::cout <<","<< 1000 * t / 1.0  <<"," <<keypoints.size();
//   std::cout << descriptorType << " descriptor,  keypoint = " <<keypoints.size()  <<" in " << 1000 * t / 1.0 << " ms" << std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // compute detector parameters based on image size
  int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = (double)cv::getTickCount();
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it)
  {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//   std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
//   std::cout<<","<<1000 * t / 1.0 <<","<<keypoints.size();
  // visualize results
  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // Detector parameters
  int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)
  double maxOverlap = 0.0;

  double t = (double)cv::getTickCount();
  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // perform a non-maximum suppression (NMS)
  for(int X_i = 0; X_i <dst_norm.rows; X_i++)
  {
    for(int Y_i = 0; Y_i < dst_norm.cols; Y_i++)
    {
      int response = (int)dst_norm.at<float>(X_i, Y_i);
      bool bOverlap = false;
      if (response > minResponse)
      {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(Y_i,X_i);
        newKeyPoint.size = 2 * apertureSize;
        newKeyPoint.response = response;

        for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
        {
          double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
          if(kptOverlap > maxOverlap)
          {
            bOverlap = true;
            if (newKeyPoint.response > (*it).response)
            {                      // if overlap is >t AND response is higher for new kpt
              *it = newKeyPoint; // replace old key point with new one
              break;             // quit loop over keypoints
            }
          }
        }
        if (!bOverlap)
        {                                          
          // only add new key point if no overlap has been found in previouS
          keypoints.push_back(newKeyPoint); 
          // store new keypoint in dynamic list
        }

      }
    }
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//   std::cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
//   std::cout<<","<<1000 * t / 1.0 <<","<<keypoints.size();

  // visualize results
  if(bVis)
  {
    std::string windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
  }

}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
  cv::Ptr<cv::FeatureDetector> detector;
  if (detectorType.compare("FAST") == 0)
  {
    int fastthreshold = 30;    // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;          // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    detector = cv::FastFeatureDetector::create(fastthreshold, bNMS, type);
  }
  if (detectorType.compare("BRISK") == 0)
  {
    int briskthresh=30;
    int octaves=3;
    float patternScale=1.0f;
    detector = cv::BRISK::create(briskthresh, octaves, patternScale);
  }
  if (detectorType.compare("ORB") == 0)
  {
    int orbnfeatures = 500;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int scoreType = (int)cv::ORB::HARRIS_SCORE;
    int patchSize = 31;      
    detector = cv::ORB::create(orbnfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K);
  }
  if (detectorType.compare("AKAZE") == 0)
  {
    cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size=0;
    int descriptor_channels=3;
    float Akazethreshold=0.001f;
    int nOctaves=4;
    int nOctaveLayers=4;
    int diffusivity = cv::KAZE::DIFF_PM_G2;
    detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, Akazethreshold, nOctaves, nOctaveLayers);
  }
  if (detectorType.compare("SIFT") == 0)
  {
    int siftnfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    detector = cv::xfeatures2d::SIFT::create(siftnfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
  }
  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//   std::cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
//   std::cout<<","<<1000 * t / 1.0 <<","<<keypoints.size();
  
  if(bVis)
  {
    std::string windowName = detectorType+" Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
  }
}
