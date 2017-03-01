/*
 * binned_detector.h
 *
 *  Created on: Oct 30, 2016
 *      Author: francisco
 */

#ifndef BINNED_DETECTOR_H_
#define BINNED_DETECTOR_H_
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

class BinnedDetector {
public:
	int bwc;//bin with col
	int bhr;//bin height row
	int MAX_COUNT;
	int brows;
	int bcols;
	int minNpoints;
	Ptr<FeatureDetector> detector;
	BinnedDetector(int phr=4,int pwc=5,int mc=50):bwc(pwc),bhr(phr),MAX_COUNT(mc),minNpoints(5){
		detector = ORB::create(MAX_COUNT);//number of features to detect
	}
	virtual ~BinnedDetector(){}
	int countPointsInBin(Mat &img,int i,int j,vector<Point2f> &pts2D){
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		Rect r(j, i, bcols, brows);
		int c=0;
		for(Point2f p:pts2D){
			if(r.contains(p)) c++;
		}
		return c;
	}
	virtual void binDetect(Mat &img,int i,int j,vector<KeyPoint> &dkpt){
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		Rect r(j, i, bcols, brows);
		Mat imgBin(img, r);
		detector->detect(imgBin, dkpt);
		//relocate
		for(unsigned int k=0;k<dkpt.size();k++){
			dkpt[k].pt+=Point2f(j,i);
		}
	}
	void refreshDetection(Mat &img,vector<Point2f> &pts2D,vector<KeyPoint> &kpts){
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		kpts.clear();
		vector<KeyPoint> dkpt;
		Mat ims;
		for(int i=0;i<img.rows-brows;i+=brows/2)
			for(int j=0;j<img.cols-bcols;j+=bcols/2){
				if(countPointsInBin(img,i,j,pts2D)<minNpoints){
					binDetect(img,i,j,dkpt);
				    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
				}
			}
	}
	void  binnedDetection(Mat &img,vector<KeyPoint> &kpts){
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		kpts.clear();
		vector<KeyPoint> dkpt;
		Mat ims;
		for(int i=0;i<img.rows-brows;i+=brows/2)
			for(int j=0;j<img.cols-bcols;j+=bcols/2){
				binDetect(img,i,j,dkpt);
			    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
			}
	}
};
class BinnedGoodFeaturesToTrack: public BinnedDetector{
public:
    TermCriteria termcrit;
    Size subPixWinSize;
	BinnedGoodFeaturesToTrack(int phr=4,int pwc=5,int mc=10):BinnedDetector(phr,pwc,mc){
		minNpoints=10;
		termcrit=TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		subPixWinSize=Size(10,10);
	}
	virtual void binDetect(Mat &img,int i,int j,vector<KeyPoint> &dkpt){
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		Rect r(j, i, bcols, brows);
		Mat imgBin(img, r);
		//detector->detect(imgBin, dkpt);
		dkpt.clear();
		vector<Point2f> points;
        cv::goodFeaturesToTrack(imgBin, points, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
        //cornerSubPix(imgBin, points, subPixWinSize, Size(-1,-1), termcrit);//give error
		//relocate
		for(unsigned int k=0;k<points.size();k++){
			Point2f p=points[k]+Point2f(j,i);
			//if(p.x>0 && p.x<img.cols & p.y>0 && p.y<img.rows)
				dkpt.push_back(KeyPoint(p,1));
		}
	}

};

#endif /* BINNED_DETECTOR_H_ */
