// Lda_face.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

int g_howManyPhotoForTraining = 8;
//每个人取出8张作为训练
int g_photoNumberOfOnePerson = 10;
//ORL数据库每个人10张图像
using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// 创建和返回一个归一化后的图像矩阵:
	Mat dst;
	switch (src.channels()) {
	case1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
//使用CSV文件去读图像和标签，主要使用stringstream和getline方法
static void readCsv(const string filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

void train() {
	string fn_csv = "imgTrainList.txt";
	vector<Mat> train_images;
	vector<int> train_labels;
	try {
		readCsv(fn_csv, train_images, train_labels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << fn_csv << ". Reason: " << e.msg << endl;
		// 文件有问题，我们啥也做不了了，退出了
		exit(1);
	}
	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->train(train_images, train_labels);
	model->save("MyFaceFisherModel.xml");
}

void test() {
	string fn_csv = "test.txt";
	vector<Mat> testImages;
	vector<int> testLabels;
	try {
		readCsv(fn_csv, testImages, testLabels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << fn_csv << ". Reason: " << e.msg << endl;
		exit(1);
	}

	string templates = "templates.txt";
	vector<Mat> templatesImages;
	vector<int> templatesLabels;
	try {
		readCsv(templates, templatesImages, templatesLabels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << templates << ". Reason: " << e.msg << endl;
		exit(1);
	}

	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->read("MyFaceFisherModel.xml");

	int iCorrectPrediction = 0;
	int predictedLabel;
	int testPhotoNumber = testImages.size();
	for (int i = 0; i < testPhotoNumber; i++) {
		predictedLabel = model->predict(testImages[i]);

		if (predictedLabel == testLabels[i])
			iCorrectPrediction++;

		string result_message = format("Test Number = %d  Actual Number = %d.", predictedLabel, testLabels[i]);
		cout << result_message << endl;

		imshow("test Object", testImages[i]);
		imshow("Dectect Object", templatesImages[testLabels[i]-1]);
		waitKey(0);
	}
	cout << "accuracy = " << float(iCorrectPrediction) / testPhotoNumber << endl;
}

int main() {
	//train();
	test();

	return 0;
}