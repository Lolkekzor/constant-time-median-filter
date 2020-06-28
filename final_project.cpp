#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> distbyte(0, 255); // distribution in range [0, 255]

using namespace std;

#define mp make_pair
#define pp pair<int, int>
#define x first
#define y second

inline int clamp(int x, int mini, int maxi) {
	if (x < mini) return mini;
	if (x > maxi) return maxi;
	return x;
}

inline int access(Mat_<uchar> m, int i, int j) {
	return m(clamp(i, 0, m.rows - 1), clamp(j, 0, m.cols - 1));
}

Mat_<uchar> median_filter_slow(Mat_<uchar> src, int kernel_size)
{
	vector<int> values;
	Mat_<uchar> dst(src.rows, src.cols);
	int offset = kernel_size / 2;
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			values.clear();
			for (int di = i - offset; di <= i + offset; di++) {
				for (int dj = j - offset; dj <= j + offset; dj++) {
					values.push_back(access(src, di, dj));
				}
			}
			nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
			dst(i, j) = values[values.size() / 2];
		}
	}

	return dst;
}

Mat_<uchar> median_filter_fast(Mat_<uchar> src, int kernel_size)
{
	Mat_<uchar> dst(src.rows, src.cols);
	vector<int> histogram(256);
	int offset = kernel_size / 2;
	int threshold = kernel_size * kernel_size / 2;
	for (int i = 0; i < dst.rows; i++) {
		for (int k = 0; k < 256; k++) histogram[k] = 0;

		if (i - offset < 0) {
			int mul = (i - offset) * -1;
			histogram[src(0, 0)] += mul * (kernel_size / 2);
		}
		if (i + offset >= dst.rows) {
			int mul = (i + offset - (dst.rows - 1));
			histogram[src(dst.rows - 1, 0)] += mul * (kernel_size / 2);
		}
		for (int di = max(0, i - offset); di <= min(dst.rows - 1, i + offset); di++) {
			histogram[src(di, 0)] += kernel_size / 2;
			for (int dj = 0; dj < kernel_size / 2; dj++) {
				if (di == 0 && i - offset < 0) {
					int mul = (i - offset) * -1;
					histogram[src(0, dj)] += mul;
				}
				if (di == dst.rows - 1 && i + offset > dst.rows - 1) {
					int mul = (i + offset - (dst.rows - 1));
					histogram[src(dst.rows - 1, dj)] += mul;
				}
				histogram[src(di, dj)]++;
			}
		}
		for (int j = 0; j < dst.cols; j++) {
			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j + offset)]++;
			}
			/// Set median
			int cnt = 0;
			for (int k = 0; k < 256; k++) {
				cnt += histogram[k];
				if (cnt > threshold) {
					dst(i, j) = k;
					break;
				}
			}

			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j - offset)]--;
			}
		}
	}

	return dst;
}


Mat_<uchar> median_filter_fast_access(Mat_<uchar> src, int kernel_size)
{
	Mat_<uchar> dst(src.rows, src.cols);
	vector<int> histogram(256);
	int offset = kernel_size / 2;
	int threshold = kernel_size * kernel_size / 2;
	for (int i = 0; i < dst.rows; i++) {
		for (int k = 0; k < 256; k++) histogram[k] = 0;
		for (int di = i - offset; di <= i + offset; di++) {
			for (int dj = -offset; dj < offset; dj++) {
				histogram[access(src, di, dj)]++;
			}
		}
		for (int j = 0; j < dst.cols; j++) {
			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j + offset)]++;
			}
			/// Set median
			int cnt = 0;
			for (int k = 0; k < 256; k++) {
				cnt += histogram[k];
				if (cnt > threshold) {
					dst(i, j) = k;
					break;
				}
			}

			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j - offset)]--;
			}
		}
	}

	return dst;
}

void median_thread(Mat_<uchar> src, Mat_<uchar> dst, int kernel_size, int start_row, int end_row) {
	vector<int> histogram(256);
	int offset = kernel_size / 2;
	int threshold = kernel_size * kernel_size / 2;
	for (int i = start_row; i < end_row; i++) {
		for (int k = 0; k < 256; k++) histogram[k] = 0;
		for (int di = i - offset; di <= i + offset; di++) {
			for (int dj = -offset; dj < offset; dj++) {
				histogram[access(src, di, dj)]++;
			}
		}
		for (int j = 0; j < dst.cols; j++) {
			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j + offset)]++;
			}
			/// Set median
			int cnt = 0;
			for (int k = 0; k < 256; k++) {
				cnt += histogram[k];
				if (cnt > threshold) {
					dst(i, j) = k;
					break;
				}
			}

			for (int di = i - offset; di <= i + offset; di++) {
				histogram[access(src, di, j - offset)]--;
			}
		}
	}

	return;
}

Mat_<uchar> median_filter_fast_multithread(Mat_<uchar> src, int kernel_size, int n_threads=8) {
	Mat_<uchar> dst(src.rows, src.cols);
	vector<int> histogram(256);
	vector<thread> threads;
	for (int t = 0; t < n_threads; t++) {
		int start_row = int(floor(double(src.rows) / n_threads * t));
		int end_row = int(floor(double(src.rows) / n_threads * (t + 1)));
		threads.push_back(thread(median_thread, src, dst, kernel_size, start_row, end_row));
	}

	for (int t = 0; t < n_threads; t++)
		threads[t].join();

	return dst;
}

Mat_<uchar> median_filter_const(Mat_<uchar> src, int kernel_size) {
	Mat_<uchar> dst(src.rows, src.cols);
	vector<vector<int> > histograms(src.cols);
	vector<int> crt_histogram(256);
	for (int k = 0; k < histograms.size(); k++)
		histograms[k].resize(256);
	int offset = kernel_size / 2;
	int threshold = kernel_size * kernel_size / 2;

	for (int j = 0; j < dst.cols; j++)
		for (int i = -offset; i < offset; i++)
			histograms[j][access(src, i, j)]++;


	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++)
			histograms[j][access(src, i + offset, j)]++;


		for (int j = -offset; j < offset; j++) {
			for (int k = 0; k < 256; k++) {
				crt_histogram[k] += histograms[max(0, j)][k];
			}
		}


		for (int j = 0; j < dst.cols; j++) {
			for (int k = 0; k < 256; k++) {
				crt_histogram[k] += histograms[min(dst.cols - 1, j + offset)][k];
			}

			/// Set median
			int cnt = 0;
			for (int k = 0; k < 256; k++) {
				cnt += crt_histogram[k];
				if (cnt > threshold) {
					dst(i, j) = k;
					break;
				}
			}

			for (int k = 0; k < 256; k++) {
				crt_histogram[k] -= histograms[max(0, j - offset)][k];
			}
		}

		for (int j = 0; j < dst.cols; j++)
			histograms[j][access(src, i - offset, j)]--;

		for (int k = 0; k < 256; k++)
			crt_histogram[k] = 0;
	}

	return dst;
}

class LeveledHistogram
{
public:
	const int IMG_DEPTH = 256, COARSE_BINS = 16, FINE_BINS = IMG_DEPTH / COARSE_BINS;
	vector<int> coarse_histogram, fine_histogram;
	LeveledHistogram() {
		fine_histogram.resize(IMG_DEPTH);
		coarse_histogram.resize(IMG_DEPTH / COARSE_BINS);
	}

	void clear() {
		for (int i = 0; i < IMG_DEPTH; i++) this->fine_histogram[i] = 0;
		for (int i = 0; i < COARSE_BINS; i++) this->coarse_histogram[i] = 0;
	}

	void operator+=(const LeveledHistogram& hist) {
		for (int k = 0; k < COARSE_BINS; k++) {
			if (hist.coarse_histogram[k] != 0) {
				for (int i = 0; i < FINE_BINS; i++)
					this->fine_histogram[k * FINE_BINS + i] += hist.fine_histogram[k * FINE_BINS + i];
				this->coarse_histogram[k] += hist.coarse_histogram[k];
			}
		}
	}

	void operator-=(const LeveledHistogram& hist) {
		for (int k = 0; k < COARSE_BINS; k++) {
			if (hist.coarse_histogram[k] != 0) {
				for (int i = 0; i < FINE_BINS; i++)
					this->fine_histogram[k * FINE_BINS + i] -= hist.fine_histogram[k * FINE_BINS + i];
				this->coarse_histogram[k] -= hist.coarse_histogram[k];
			}
		}
	}

	void incrementPosition(int position) {
		this->fine_histogram[position]++;
		this->coarse_histogram[position / FINE_BINS]++;
	}

	void decrementPosition(int position) {
		this->fine_histogram[position]--;
		this->coarse_histogram[position / FINE_BINS]--;
	}

	int getMedian(int threshold) {
		int elems = 0;
		int coarse_pos = 0, fine_pos = 0;
		while (elems + this->coarse_histogram[coarse_pos] <= threshold) {
			elems += this->coarse_histogram[coarse_pos];
			coarse_pos++;
		}
		while (elems + this->fine_histogram[coarse_pos * FINE_BINS + fine_pos] <= threshold) {
			elems += this->fine_histogram[coarse_pos * FINE_BINS + fine_pos];
			fine_pos++;
		}
		return coarse_pos * FINE_BINS + fine_pos;
	}
};

Mat_<uchar> median_filter_leveled(Mat_<uchar> src, int kernel_size) {
	Mat_<uchar> dst(src.rows, src.cols);
	vector<LeveledHistogram > histograms(src.cols);
	LeveledHistogram crt_histogram;
	int offset = kernel_size / 2;
	int threshold = kernel_size * kernel_size / 2;

	for (int j = 0; j < dst.cols; j++)
		for (int i = -offset; i < offset; i++)
			histograms[j].incrementPosition(access(src, i, j));


	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++)
			histograms[j].incrementPosition(access(src, i + offset, j));


		for (int j = -offset; j < offset; j++) {
			crt_histogram += histograms[max(0, j)];
		}


		for (int j = 0; j < dst.cols; j++) {
			crt_histogram += histograms[min(dst.cols - 1, j + offset)];

			/// Set median
			dst(i, j) = crt_histogram.getMedian(threshold);

			crt_histogram -= histograms[max(0, j - offset)];
		}

		for (int j = 0; j < dst.cols; j++)
			histograms[j].decrementPosition(access(src, i - offset, j));

		crt_histogram.clear();
	}

	return dst;
}

void benchmark(char fname[]) {
	Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	ofstream data;
	data.open("benchmark_largeimg_extraksize.csv");

	const int NR_REPS = 5;
	for (int ksize = 103; ksize <= 251; ksize += 2) {
		std::chrono::duration<double> elapsed_cv, elapsed_const, elapsed_leveled;
		for (int reps = 0; reps < NR_REPS; reps++) {
			cout << "Benchmarking... Ksize {" << ksize << "}, Rep {" << reps << "}\n";

			auto start = std::chrono::high_resolution_clock::now();
			Mat_<uchar> blur_opencv;
			medianBlur(src, blur_opencv, ksize);
			auto finish = std::chrono::high_resolution_clock::now();
			elapsed_cv += finish - start;

			start = std::chrono::high_resolution_clock::now();
			Mat_<uchar> blur_const = median_filter_const(src, ksize);
			finish = std::chrono::high_resolution_clock::now();
			elapsed_const += finish - start;

			start = std::chrono::high_resolution_clock::now();
			Mat_<uchar> blur_leveled = median_filter_leveled(src, ksize);
			finish = std::chrono::high_resolution_clock::now();
			elapsed_leveled += finish - start;
		}

		elapsed_cv /= NR_REPS;
		elapsed_const /= NR_REPS;
		elapsed_leveled /= NR_REPS;

		data 
			<< elapsed_cv.count() << ',' 
			<< elapsed_const.count() << ',' 
			<< elapsed_leveled.count() << '\n';
	}
}

int main()
{
	destroyAllWindows();

	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> blur_opencv, blur_slow, blur_fast, blur_fast_access, blur_const, blur_multithread, blur_leveled;

	int ksize;
	cout << "Enter kernel size: ";
	cin >> ksize;


	//benchmark(fname);

	auto start = std::chrono::high_resolution_clock::now();
	medianBlur(src, blur_opencv, ksize);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_cv = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_slow = median_filter_slow(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_slow = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_fast = median_filter_fast(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_fast = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_fast_access = median_filter_fast_access(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_fast_access = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_multithread = median_filter_fast_multithread(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_fast_multithread = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_const = median_filter_const(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_const = finish - start;

	start = std::chrono::high_resolution_clock::now();
	blur_leveled = median_filter_leveled(src, ksize);
	finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_leveled = finish - start;

	imshow("IMG", src);
	imshow("BLUR OPENCV", blur_opencv);
	imshow("BLUR SLOW", blur_slow);
	imshow("BLUR FAST", blur_fast);
	imshow("BLUR CONST", blur_const);
	imshow("BLUR MULTITHREAD", blur_multithread);
	imshow("BLUR LEVELED", blur_leveled);

	cout << "OpenCV runtime: " << elapsed_cv.count() << '\n';
	cout << "Slow runtime: " << elapsed_slow.count() << '\n';
	cout << "Fast runtime: " << elapsed_fast.count() << '\n';
	cout << "Fast accesss: " << elapsed_fast_access.count() << '\n';
	cout << "Fast multithread: " << elapsed_fast_multithread.count() << '\n';
 	cout << "Const runtime: " << elapsed_const.count() << '\n';
	cout << "Leveled runtime: " << elapsed_leveled.count() << '\n';

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (blur_fast_access(i, j) != blur_fast(i, j)) {
				cout << "DIFFERENCE";
			}
			if (blur_slow(i, j) != blur_opencv(i, j)) {
				cout << "DIFFERENCE";
			}
			if (blur_slow(i, j) != blur_const(i, j)) {
				cout << "DIFFERENCE";
			}
			if (blur_leveled(i, j) != blur_const(i, j)) {
				cout << "DIFFERENCE";
			}
		}
	}
	waitKey(0); 
}
