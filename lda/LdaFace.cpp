// Lda_face.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
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

static Mat norm_0_255(String _src) {

	Mat src = imread(_src, 0);
	// �����ͷ���һ����һ�����ͼ�����:
	Mat dst;
	switch (src.channels()) {
	case1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
//ʹ��CSV�ļ�ȥ��ͼ��ͱ�ǩ����Ҫʹ��stringstream��getline����
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
			images.push_back(norm_0_255(path));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

static void readTest(const string filename, vector<String>& images, vector<int>& labels, char separator = ';') {
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
			images.push_back(path);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

void train(string imgTrainList, string faceModelPath) {
	vector<Mat> train_images;
	vector<int> train_labels;
	try {
		readCsv(imgTrainList, train_images, train_labels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << imgTrainList << ". Reason: " << e.msg << endl;
		// �ļ������⣬����ɶҲ�������ˣ��˳���
		exit(1);
	}
	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->train(train_images, train_labels);
	model->save(faceModelPath + "MyFaceFisherModel.xml");
}

int predict(string imagePath, string faceModelPath) {
	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->read(faceModelPath + "MyFaceFisherModel.xml");
	Mat img = norm_0_255(imagePath);
	int predictedLabel = model->predict(img);
	return predictedLabel;
}

void test(string testFile, string templates, string faceModelPath) {
	vector<String> testImages;
	vector<int> testLabels;
	try {
		readTest(testFile, testImages, testLabels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << testFile << ". Reason: " << e.msg << endl;
		exit(1);
	}

	vector<Mat> templatesImages;
	vector<int> templatesLabels;
	try {
		readCsv(templates, templatesImages, templatesLabels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << templates << ". Reason: " << e.msg << endl;
		exit(1);
	}

	int iCorrectPrediction = 0;
	int predictedLabel;
	int testPhotoNumber = testImages.size();
	for (int i = 0; i < testPhotoNumber; i++) {

		predictedLabel = predict(testImages[i], faceModelPath);

		if (predictedLabel == testLabels[i]) {
			iCorrectPrediction++;
		}

		string result_message = format("Test Number = %d  Actual Number = %d.", predictedLabel, testLabels[i]);
		cout << result_message << endl;

		imshow("test Object", imread(testImages[i]));
		int index = testLabels[i] - 1;
		imshow("Dectect Object", templatesImages[index]);
		waitKey(0);
	}
	cout << "accuracy = " << float(iCorrectPrediction) / testPhotoNumber << endl;
}



int main() {

	string imgTrainList = "data/imgTrainList.txt";
	String faceModelPath = "data/";
	train(imgTrainList,faceModelPath);

	string testFile = "data/test.txt";
	string templates = "data/templates.txt";
	test(testFile, templates, faceModelPath);

	return 0;
}