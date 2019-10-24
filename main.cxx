////////////////////////////////////////////////////////////////////
/// At first you must set stack commit and stack resservd to 78125000
////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include <chrono> 
#include<string> 
#include<math.h>
#include<fstream>

using namespace cv;
using namespace std;
using namespace std::chrono;
struct pixel
{
	int row;
	int column;
	int disparity;
	bool consistance;
};
struct Stain
{
	int i;  // center of aera of stain in x direction.
	int j;  // center of aera of stain in y direction.
	int minI; // boundry of stain in x direction.
	int maxI;
	int minJ; // boundry of stain in y direction.
	int maxJ;
	int area=0;
	vector<Point> stainPoints;

};
struct stainLimit
{
	int startingRow = 0;
	int endingRow = 0;
};
typedef vector<shared_ptr<stainLimit>> stainLimits;
int numOfColumns;
int numOfRows;
int numOfColumnsResized;
int numOfRowsResized;
int thickness = 60;
int maxDisparity = 30;
int maxkernelSize = 35; // kernel size must be odd number.
int kernelSize = 9;
auto sstereoResult = make_shared<Mat>();
typedef vector<pixel*> layerVector;
vector<layerVector> layers;
void ReadBothImages(shared_ptr<Mat>, shared_ptr<Mat>);
void Meshing(int, int, int, int, int);
double CalcDistance(int, int, int, int);
int CalcCost(shared_ptr<Mat>, shared_ptr<Mat>, int, int, int, int,int);
Vec3b bgrPixel_02(0, 255, 255);
Vec3b bgrPixel_04(255, 0, 0);
Vec3b bgrPixel_03(0, 255, 0);
Vec3b bgrPixel_01(0, 0, 255);
Vec3b bgrBackground(0, 0, 0);
vector<int*> stainSize;
void SSDstereo(shared_ptr<Mat>, shared_ptr<Mat>,shared_ptr<Mat>, int, int,int,int);
//void selsectiveStereo(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, layerVector*, int, int);
stainLimits selsectiveStereo(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, int, int,int, int);
void prepareResult(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, vector<layerVector>, int, int, int);
void reportResult(string );
void filterResult(shared_ptr<Mat>, shared_ptr<Mat>, Vec3b);
void checkPoint(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Stain>, int, int, Vec3b,int*);
void makeStain(shared_ptr<Mat> , shared_ptr<Mat> , shared_ptr<Stain> , int , int, Vec3b,int*);
void stainDetector(shared_ptr<Mat>, shared_ptr<Mat>, Vec3b, shared_ptr<vector<shared_ptr<Stain>>>);
void mergingStains(shared_ptr<Mat> , shared_ptr<vector<shared_ptr<Stain>>> );
int main()
{
	
	shared_ptr<Mat> rightImage= make_shared<Mat>() ;
	shared_ptr<Mat> leftImage=make_shared<Mat>();
	shared_ptr<Mat> rightImageResized = make_shared<Mat>();
	shared_ptr<Mat> leftImageResized = make_shared<Mat>();
	
	shared_ptr<Mat>  stereoResut= make_shared<Mat>();
	shared_ptr<Mat>  stereoResutResized= make_shared<Mat>();

	auto start = chrono::high_resolution_clock::now();

	ReadBothImages(leftImage, rightImage);
	//imshow("test", *leftImage);
	//waitKey(0);
	numOfRows = leftImage->rows;
	numOfColumns = leftImage->cols;
	stereoResut = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	SSDstereo(leftImage, rightImage, stereoResut, kernelSize, maxDisparity, numOfRows, numOfColumns);
	cv::imshow("stereoOutput", *stereoResut);
	cv::waitKey(1000);
	chrono::high_resolution_clock::time_point stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	auto value = duration.count();
	string duration_s = to_string(value);
	ofstream repotredResult;
	repotredResult.open("result.txt");
	repotredResult << "Totaltime of SSDstereo result is " + duration_s + " (s)" << std::endl;
	repotredResult.close();


		try
		{
	}
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

	//auto start = chrono::high_resolution_clock::now();
	cv::resize(*rightImage, *rightImageResized, cv::Size(), 0.5, 0.5);
	cv::resize(*leftImage, *leftImageResized, cv::Size(), 0.5, 0.5);
	numOfRowsResized = leftImageResized->rows;
	numOfColumnsResized = leftImageResized->cols;
	stereoResutResized = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	SSDstereo(leftImageResized, rightImageResized, stereoResutResized, kernelSize, maxDisparity, numOfRowsResized, numOfColumnsResized);
	cv::imshow("stereoOutputResized", *stereoResutResized);
	cv::waitKey(10000);
	//chrono::high_resolution_clock::time_point stop = high_resolution_clock::now();
	//auto duration = duration_cast<seconds>(stop - start);
	//auto value = duration.count();
	//string duration_s = to_string(value);
	//ofstream repotredResult;
	//repotredResult.open("result.txt", std::ios_base::app | std::ios_base::out);
	//repotredResult << "Totaltime of SSDstereo result for resized is " + duration_s + " (s)" << std::endl;
	//repotredResult.close();

	try
	{}
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
	



	for (int midDis = 1; midDis <= maxDisparity; midDis++) {
		auto result_00 = make_shared<Mat>(numOfRowsResized, numOfColumnsResized, CV_8UC1);// Stereo result.
		auto result_01 = make_shared<Mat>(numOfRowsResized, numOfColumnsResized, CV_8UC3);// Selective stereo L2R.
		auto result_02 = make_shared<Mat>(numOfRowsResized, numOfColumnsResized, CV_8UC3);// Selective stereo R2L.
		auto result_03 = make_shared<Mat>(numOfRowsResized, numOfColumnsResized, CV_8UC3);// slective with L2R and R2L consistance.
		auto result_04 = make_shared<Mat>(numOfRowsResized, numOfColumnsResized, CV_8UC3);// slective with L2R and R2L notconsistance.
		*result_00 = *stereoResutResized;
		cvtColor(*result_00, *result_01, CV_GRAY2RGB);
		cvtColor(*result_00, *result_02, CV_GRAY2RGB);
		cvtColor(*result_00, *result_03, CV_GRAY2RGB);
		////////////////////////////////////////////////////////////////////
		/// In this part we have impelemet selective stereo.
		////////////////////////////////////////////////////////////////////
		
		stainLimits result_sainLimitsResized = selsectiveStereo(leftImageResized, rightImageResized, result_01, result_02, result_03, kernelSize, midDis, numOfRowsResized, numOfColumnsResized);
		std::cout << "the midDisparity is " << midDis << std::endl;
		std::cout << "the result_sainLimitsResized.size is " << result_sainLimitsResized.size() << std::endl; 
		for (int q = 0; q < result_sainLimitsResized.size(); q++) {
			std::cout << "the stain limit is " << result_sainLimitsResized[q]->startingRow <<" ----" << result_sainLimitsResized[q]->endingRow << std::endl;
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////
/// In this part we can load two Images.
////////////////////////////////////////////////////////////////////
void ReadBothImages(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage) {
	try {

		*rightImage = imread("1.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the right image
		//rightImage->convertTo(*rightImage, CV_64F);
		*leftImage = imread("2.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the left image
		//leftImage->convertTo(*leftImage, CV_64F);
	}
	catch (char* error) {
		cout << "can not load the " << error << " iamge" << endl;
	}
	//imshow("test", *rightImage);

	//waitKey(0);
}


////////////////////////////////////////////////////////////////////
/// In this part we clac layer of each pixel.
////////////////////////////////////////////////////////////////////
void Meshing(int numOfRows, int numOfColumns, int thickness, int kernelSize, int maxDisparity) {
	int tempLayer = 0;
	int numOfLayers = int(CalcDistance(numOfRows, numOfColumns, 0, 0) / thickness);
	// the number 4 thai wrote there is for ensure that all of the image has suported... dont wworry... we have delete those who is null.
	for (int i = 1; i <= numOfLayers + 4; i++) {
		layerVector tempLayer;
		layers.push_back(tempLayer);
	}
	for (int i = (kernelSize / 2); i < numOfRows - (kernelSize / 2); i++) {
		for (int j = (kernelSize / 2); j < numOfColumns - (kernelSize / 2) - maxDisparity; j++) {
			tempLayer = int(CalcDistance(numOfRows, numOfColumns, i, j) / thickness);
			pixel* tempLocation = new pixel;
			tempLocation->row = i;
			tempLocation->column = j;
			layers.at(tempLayer).push_back(tempLocation);
		}
	}
	// this part is added to avoid vector with zeero size.
	for (int i = layers.size() - 1; i >= 0; i = i - 1) {
		if (layers[i].size() == 0) {
			layers.erase(layers.begin() + i);
		}
	}
}


////////////////////////////////////////////////////////////////////
/// In this part we clac distance of each pixel.
////////////////////////////////////////////////////////////////////
double CalcDistance(int numOfRows, int numOfColumns, int row, int column) {
	double tempDistance = sqrt(pow((row - numOfRows), 2) + pow((column - int(numOfColumns / 2) + .05), 2));
	return tempDistance;
}


////////////////////////////////////////////////////////////////////
/// In this part we clac disparity of each pixel.
////////////////////////////////////////////////////////////////////
void  SSDstereo(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_,shared_ptr<Mat> result_temp_, int kernelSize, int maxDisparity,int NRow,int NCols) {
	int tempCost = 0;
	int tempDisparity = 0;
	
	for (int u = (kernelSize/2)+1; u <(NCols -maxDisparity-kernelSize/2)-1 ; u++) {
		for (int v = (kernelSize / 2) + 1; v <NRow-(kernelSize / 2); v++) {
			double cost = 10000000;
			tempCost = 0;
			tempDisparity = 0;
			for (int i = 0; i < maxDisparity; i++) {
				tempCost = CalcCost(leftImage_, rightImage_,v ,u , kernelSize, i, NCols);
				if (tempCost < cost) {
					cost = tempCost;
					tempDisparity = i;
				}
			}
			tempDisparity = tempDisparity * 255 / maxDisparity;
			result_temp_->at<uchar>(v, u) = tempDisparity;
			//std::cout << " tempDisparity for ("<< u<<","<<v<<") is "  << tempDisparity << std::endl;
		}
	}
	//std::cout << "debug" << std::endl;
	//cv::imshow("stereoOutput", *result_temp);
	//cv::waitKey(100);
}


////////////////////////////////////////////////////////////////////
/// In this part we clac cost of each pixel for sepecfic disparity.
////////////////////////////////////////////////////////////////////
int CalcCost(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, int row, int column, int kernelSize, int disparity,int NCols) {
	int cost = 0;
	for (int u = -int(kernelSize / 2); u <= int(kernelSize / 2); u++) {
		for (int v = -int(kernelSize / 2); v <= int(kernelSize / 2); v++) {
			int temp1 = row + u;
			int temp2 = column + v;
			int temp3 = row + u + disparity;
			int temp4 = column + v;
			// for error handeling.
			if (column + u + disparity >= NCols) {
				cout << "*****************************************************" << endl;
			}
			cost = cost + int(pow((leftImage_->at<uchar>(row + v, column + u) - (rightImage_->at<uchar>(row + v, column + u + disparity))), 2));
		}
	}
	return cost;
}


////////////////////////////////////////////////////////////////////
/// In this part we clac selective disparity of each pixel.
////////////////////////////////////////////////////////////////////

stainLimits selsectiveStereo(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, shared_ptr<Mat> result_1_, shared_ptr<Mat> result_2_, shared_ptr<Mat> result_3_, int kernelSize, int midelDisparity, int NRow, int NCols) {
	auto start = chrono::high_resolution_clock::now();
	bool left2right = false;
	bool right2let = false;
	int temp0 = midelDisparity - 1;
	int temp1 = midelDisparity;
	int temp2 = midelDisparity + 1;

	int temp3 = -(midelDisparity - 1);
	int temp4 = -midelDisparity;
	int temp5 = -(midelDisparity + 1);

	int tempCost0;
	int tempCost1;
	int tempCost2;

	int tempCost3;
	int tempCost4;
	int tempCost5;
	int numOfDetectedPixel;
	bool triger = false;
	stainLimits tempStainLimits;
	bool creationTriger = false;
	shared_ptr <stainLimit > tempStainLimit ;
	for (int v = (kernelSize / 2) + 10; v < NRow-(kernelSize / 2)-10; v++) {
		numOfDetectedPixel = 0;
		

		for (int u = (kernelSize / 2) + 1; u < (NCols - maxDisparity - kernelSize / 2) - 1; u++) {
			left2right = false;
			right2let = false;
			tempCost0 = CalcCost(leftImage_, rightImage_, v, u, kernelSize, temp0, NCols);
			tempCost1 = CalcCost(leftImage_, rightImage_, v, u, kernelSize, temp1, NCols);
			tempCost2 = CalcCost(leftImage_, rightImage_, v, u, kernelSize, temp2, NCols);

			tempCost3 = CalcCost(rightImage_, leftImage_, v, u, kernelSize, temp3, NCols);
			tempCost4 = CalcCost(rightImage_, leftImage_, v, u, kernelSize, temp4, NCols);
			tempCost5 = CalcCost(rightImage_, leftImage_, v, u, kernelSize, temp5, NCols);
			

			/////////////////////////////////////////////////////////////////////
			///////////// In these two if cluse we are making image by left ref and right ref ... and after that we are making crose checked redult.
			////////////////////////////////////////////////////////////////////

			if (tempCost1 < tempCost0 & tempCost1 < tempCost2) {
				left2right = true;
				//result_1->at<Vec3b>(Point(u, v)) = bgrPixel_01;
			}
			if (tempCost4 < tempCost3 & tempCost4 < tempCost5) {
				right2let = true;
				//result_2->at<Vec3b>(Point(u, v)) = bgrPixel_02;
			}
			if ((left2right & right2let)) {
				numOfDetectedPixel = numOfDetectedPixel + 1;
				result_3_->at<Vec3b>(Point(u, v)) = bgrPixel_04;
			}
		}
		if (!creationTriger) {
			auto tempTemp = make_shared<stainLimit>();
			tempStainLimit = tempTemp;
			creationTriger = true;
		}
		if (!triger &( numOfDetectedPixel > NCols / 3)) {
			triger = true;
			//std::cout << "the starting  v is " << v << std::endl;
			tempStainLimit->startingRow = v;
			//std::cout << "the starting  v in the vector is " << tempStainLimit->startingRow << std::endl;

		}
		if( (triger & numOfDetectedPixel < (NCols / 4))| (triger & v==(NCols - maxDisparity - kernelSize / 2) - 2)) {
			tempStainLimit->endingRow = v;
			//std::cout << "the starting  v in the vector is " << tempStainLimit->startingRow << std::endl;
			//std::cout << "the ending  v is " << v << std::endl;
			//std::cout << "the ending  v in the vector is " << tempStainLimit->endingRow << std::endl;
			triger = false;
			creationTriger = false;
			tempStainLimits.push_back(tempStainLimit);
		}


	}
	//auto totalResult = make_shared<Mat>(NRow, 2* NCols, CV_8UC3);;
	//*totalResult = *result_1_;
	//imshow("result_1", *result_1);
	//imshow("result_2", *result_2);
	imshow("result_3_", *result_3_);
	//cv::hconcat(*result_1_, *result_2_, *totalResult);
	//cv::hconcat(*totalResult, *result_3_, *totalResult);
	string tempName = "totalResult_midDisparity_" + to_string(midelDisparity) + ".png";
	//imshow(tempName, *totalResult);
	cv::waitKey(1);

	chrono::high_resolution_clock::time_point stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	auto value = duration.count();
	string duration_s = to_string(value);
	string TempTimer = "the time of calculation of "+to_string(midelDisparity)+" midelDisparity is "+ duration_s + " (s)";
	reportResult(TempTimer);
	return tempStainLimits;

	
		
}

////////////////////////////////////////////////////////////////////
/// In this part we will Filter the result.
////////////////////////////////////////////////////////////////////
void filterResult(shared_ptr<Mat> background, shared_ptr<Mat> input, Vec3b Color) {
	int numberOfHorizontalChecker = 1;
	bool tempCorrect = false; // it means this picxel is not corecctly selected.
	for (int i = 0; i < numOfRows; i++) {
		for (int j = numberOfHorizontalChecker; j < numOfColumns - numberOfHorizontalChecker; j++) {
			if (input->at<Vec3b>(Point(j, i)) == Color) {
				tempCorrect = false;
				for (int k = 1; k <= numberOfHorizontalChecker; k++) {
					if (input->at<Vec3b>(Point(j + k, i)) == Color | input->at<Vec3b>(Point(j - k, i)) == Color) {
						tempCorrect = true;
						break;
					}
				}
				if (!tempCorrect) {
					input->at<Vec3b>(Point(j, i)) = background->at<Vec3b>(Point(j, i));
				}
			}
		}
	}


}


////////////////////////////////////////////////////////////////////
/// In this part we will detcte the stain.
////////////////////////////////////////////////////////////////////
void stainDetector(shared_ptr<Mat> background, shared_ptr<Mat> input, Vec3b Color,shared_ptr<vector<shared_ptr<Stain>>> stainResults) {
	
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			if (input->at<Vec3b>(Point(i, j)) == Color) {
				std::cout << "(" << i << "," << j << ")" << endl;
				shared_ptr<Stain> stain_temp = make_shared<Stain> ();
				stain_temp->minI = i;
				stain_temp->maxI = i;
				stain_temp->minJ = j;
				stain_temp->maxJ = j;
				auto alpha = new int(0);
				std::cout << "this Stain is called " << std::endl;
				makeStain(background, input, stain_temp, i, j, Color, alpha);
				std::cout << "make stain is done." << std::endl;
				if (stain_temp->area >= 1) {   //in this part we will control removing stains by area of stain. 
					stainResults->push_back(stain_temp);
					stainSize.push_back(alpha);
				}
			}

		}
	}
}

void makeStain(shared_ptr<Mat> background, shared_ptr<Mat> input, shared_ptr<Stain> stain, int i, int j,Vec3b Color, int* alpha) {
	std::cout << "Debug in stain" << std::endl;

	checkPoint(background, input,stain, i, j, Color, alpha);
}

void checkPoint(shared_ptr<Mat> background, shared_ptr<Mat> input, shared_ptr<Stain> stain, int i, int j, Vec3b Color,int* alpha) {
	//std:: cout<< "Debug in checkPoint" << std::endl;
	//std::cout << "i is->(" << j << ")" << endl;
	//std::cout << "j is->(" << i << ")" << endl;
	if (input->at<Vec3b>(Point(i, j)) == Color) {
		
		input->at<Vec3b>(Point(i, j)) = background->at<Vec3b>(Point(i, j));
		stain->stainPoints.push_back(Point(i, j));
		stain->area = stain->area + 1;
		if (i > stain->maxI) {
			stain->maxI = i;
		}
		if (j > stain->maxJ) {
			stain->maxJ = j;
		}
		(*alpha) = (*alpha) + 1;
		
		if (stain->stainPoints.back().x == i & stain->stainPoints.back().y == j) {
			//std::cout << "debug 00001" << std::endl;
			checkPoint(background, input, stain, i + 1, j, Color,alpha);
//			std::cout << "debug 00002" << std::endl;
			checkPoint(background, input, stain, i, j + 1, Color, alpha);
			//std::cout << "debug 00003" << std::endl;
			checkPoint(background, input, stain, i + 1, j + 1, Color, alpha);
			//std::cout << "debug 00004" << std::endl;
			checkPoint(background, input, stain, i - 1, j, Color, alpha);
			//std::cout << "debug 00005" << std::endl;
			checkPoint(background, input, stain, i, j - 1, Color, alpha);
			//std::cout << "debug 00006" << std::endl;
			checkPoint(background, input, stain, i - 1, j - 1, Color, alpha);
		}
	}
}


////////////////////////////////////////////////////////////////////
/// In this part we will merge the stain.
////////////////////////////////////////////////////////////////////
void mergingStains(shared_ptr<Mat> input, shared_ptr<vector<shared_ptr<Stain>>> staingResults) {

	vector<shared_ptr<Stain>>::iterator ptr1;
	vector<shared_ptr<Stain>>::iterator ptr2;
	vector<int> distances;
	int distanceX;
	int distanceY;
	for (ptr1 = (*staingResults).begin(); ptr1 < (*staingResults).end(); ptr1++) {
		for (ptr2 = (*staingResults).begin(); ptr2 < (*staingResults).end(); ptr2++) {
			if (ptr1 == ptr2) {
				continue;
			}
			distances.clear();
			distances.push_back(abs((*ptr1)->minI - (*ptr2)->minI));
			distances.push_back(abs((*ptr1)->minI - (*ptr2)->maxI));
			distances.push_back(abs((*ptr1)->maxI - (*ptr2)->minI));
			distances.push_back(abs((*ptr1)->maxI - (*ptr2)->maxI));
			distances.push_back(abs((*ptr1)->minJ - (*ptr2)->minJ));
			distances.push_back(abs((*ptr1)->minJ - (*ptr2)->maxJ));
			distances.push_back(abs((*ptr1)->maxJ - (*ptr2)->minJ));
			distances.push_back(abs((*ptr1)->maxJ - (*ptr2)->maxJ));
			int distance = 100000000;

			distanceX = std::min(std::min(distances[0], distances[1]), std::min(distances[2], distances[3]));
			distanceY = std::min(std::min(distances[4], distances[5]), std::min(distances[6], distances[7]));
			for (vector<int>::iterator it = distances.begin(); it != distances.end(); it++) {
				if ((*it) <= distance) {
					distance = (*it);
				}
			}
			//cout << "***************************************************************************" << endl;
			//cout << "the distance is " << distance << "." << endl;
			if (distanceX<40 & distanceY<10) {
				int mergingRectWidth = std::max(std::max(distances[0], distances[1]), std::max(distances[2], distances[3]));
				int mergingRectHight = std::max(std::max(distances[4], distances[5]), std::max(distances[6], distances[7]));
				cv::Rect addingRect = Rect(std::min((*ptr1)->minI, (*ptr2)->minI), std::min((*ptr1)->minJ, (*ptr2)->minJ), mergingRectWidth, mergingRectHight);
				cv::rectangle(*input, addingRect, cv::Scalar(0, 0, 255));
			}
		}
	}
}



void reportResult(string Text) {

	ofstream repotredResult;
	repotredResult.open("result.txt", std::ios_base::app | std::ios_base::out);
	repotredResult << Text << std::endl;
	repotredResult.close();


}