#include "Filter.h"
#include "glm/glm.hpp"


#include <queue>
#include <random>
#include <limits>
#include <cmath>


static inline float clampf(float v, float lo, float hi) {
	return (v < lo) ? lo : ((v > hi) ? hi : v);
}

Filter::Filter() {}

Filter::~Filter() {}


void Filter::colorToGrayscale(cv::Mat colorImage) {
	int width = colorImage.cols;
	int height = colorImage.rows;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = colorImage.at<cv::Vec4b>(y, x);
			uchar gray = (uchar)((float)color[0] * 0.114 + (float)color[1] * 0.587 + (float)color[2] * 0.299);
			colorImage.at<cv::Vec4b>(y, x) = cv::Vec4b(gray, gray, gray, color[3]);
		}
	}
}


void Filter::colorToGrayscale(cv::Vec4b* colorData, int width, int height) {
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//offset = y * width + x;
			cv::Vec4b color = colorData[offset];
			uchar gray = (color[0] + color[1] + color[2]) / 3;
			colorData[offset] = cv::Vec4b(gray, gray, gray, color[3]);
			offset++;
		}
	}
}

void Filter::filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height) {
	int offset, offset_neighbor;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			cv::Vec4f color = cv::Vec4f(0, 0, 0, 0);
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b color_neighbor = colorData[offset_neighbor];
					color += (cv::Vec4f)color_neighbor;
				}
			}

			color /= 9;
			offset = y * width + x;
			colorProcessedData[offset] = cv::Vec4b(color[0], color[1], color[2], color[3]);

		}
	}

}

void Filter::filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	uchar* gaussianKernel = new uchar[25]{
		1, 4, 7, 4, 1,
		4, 16, 26, 16, 4,
		7, 26, 41, 26, 7,
		4, 16, 26, 16, 4,
		1, 4, 7, 4, 1
	};
	std::memcpy(depthProcessedData, depthData, width * height * sizeof(cv::Vec4b));
	// Initally, just copy the original data in here.
	auto offset = 0;
	// As I have value ,,1" in the corners, I will begin with 2 fors.
	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {
			offset = y * width + x;
			// to know exactly where I am in the image
			// take the pixels around the current pixel (x, y)
			int sum = 0;
			for (int k = -2; k <= 2; k++) {
				for (int l = -2; l <= 2; l++) {
					// above and below the current pixel, left and right.
					int offset_element = (y + k) * width + (x + l);
					int val_adancime = depthData[offset_element][0];;
					sum += val_adancime * gaussianKernel[(k + 2) * 5 + (l + 2)];
				}
			}
			// the average but divided by the sum of the kernel values.
			float filtered_f = static_cast<float>(sum) / 273.0f;
			uchar new_pixel_value = static_cast<uchar>(filtered_f);
			depthProcessedData[offset] = cv::Vec4b(new_pixel_value, new_pixel_value, new_pixel_value, depthData[offset][3]);
		}
	}
}

void Filter::filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	uchar* gaussianKernel = new uchar[9]{
		1, 2, 1,
		2, 4, 2,
		1, 2, 1
	};
	// Initally, just copy the original data in here.
	std::memcpy(grayscaleProcessedData, grayscaleData, static_cast<size_t>(width) * height * sizeof(uchar));
	auto offset = 0;
	// As I have value ,,1" in the corners, I will begin with 2 fors.
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x;
			// to know exactly where I am in the image
			// take the pixels around the current pixel (x, y)
			int sum = 0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					// above and below the current pixel, left and right.
					int offset_element = (y + k) * width + (x + l);
					sum += grayscaleData[offset_element] * gaussianKernel[(k + 1) * 3 + (l + 1)];
				}
			}
			// And I replace the pixel value with the average of the values in his neighborhood.
			uchar val_finala_pixel = static_cast<uchar>(sum / 16);
			grayscaleProcessedData[offset] = (uchar)val_finala_pixel;
		}
	}

}

void Filter::filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, float* depthMeasureData, int width, int height) {
	std::memcpy(depthProcessedData, depthData, width * height * sizeof(cv::Vec4b));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = y * width + x;
			float d = depthMeasureData[offset];
			uchar out;
			// discretization of the depth values into the 5 intervals.
			if (d < 0.5f) {
				out = 51;
			}
			else if (d < 1.0f) {
				out = 102;
			}
			else if (d < 1.5f) {
				out = 153;
			}
			else if (d < 2.0f) {
				out = 204;
			}
			else {
				out = 255;
			}
			depthProcessedData[offset] = cv::Vec4b(out, out, out, depthData[offset][3]);
		}
	}
}

// after applying this median filter, the grayscaleProcessedData will contain the processed image.
void Filter::filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset = 0;
	// median array with 9 elements ( 3 * 3)
	// Initally, just copy the original data
	memcpy(grayscaleProcessedData, grayscaleData, static_cast<size_t>(width) * height * sizeof(uchar));
	// into the processed data array.
	std::vector<int> median_elements(10);
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x; // to know exactly where I am in the image
			// take the pixels around the current pixel (x, y)
			int i = 0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					// above and below the current pixel, left and right.
					int offset_neigh = (y + k) * width + (x + l);
					median_elements.at(i++) = grayscaleData[offset_neigh];
				}
			}
			// I sort the elements in the median_elements vector and choose the median value.
			auto m = median_elements.begin() + median_elements.size() / 2;
			std::nth_element(median_elements.begin(), m, median_elements.end());
			auto valoarea_mediana = median_elements.at(median_elements.size() / 2);
			grayscaleProcessedData[offset] = valoarea_mediana;
		}
	}
}

void Filter::filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	// the sobel operators.
	int* sobelM_x = new int[9] {
		-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1
		};
	int* sobelM_y = new int[9] {
		-1, -2, -1,
			0, 0, 0,
			1, 2, 1
		};
	std::memcpy(grayscaleProcessedData, grayscaleData, width * height * sizeof(uchar));

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int offset = y * width + x;
			int sum_x = 0, sum_y = 0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					int offset_el = (y + k) * width + (x + l);
					uchar val_din_fereastra = grayscaleData[offset_el];
					sum_x += val_din_fereastra * sobelM_x[(k + 1) * 3 + (l + 1)];
					sum_y += val_din_fereastra * sobelM_y[(k + 1) * 3 + (l + 1)];
				}
			}
			// I compute the magnitude of the gradient
			int magnitude = std::sqrt(std::pow(std::abs(sum_x), 2) + std::pow(std::abs(sum_y), 2));
			// if the magnitude is greater than 255, i will truncate it to 255
			// and display the value in the grayscale processed image.
			grayscaleProcessedData[offset] = (uchar)std::min(magnitude, 255);
		}
	}
}

void Filter::filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	// almost the same as the sobel filter, but with different masks.
	int* prewittM_x = new int[9] {
		1, 0, -1,
			1, 0, -1,
			1, 0, -1
		};
	int* prewittM_y = new int[9] {
		1, 1, 1,
			0, 0, 0,
			-1, -1, -1
		};
	std::memcpy(depthProcessedData, depthData, width * height * sizeof(cv::Vec4b));
	auto offset = 0;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x;
			auto sum_x = 0;
			auto sum_y = 0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					int offset_el = (y + k) * width + (x + l);
					sum_x += depthData[offset_el][0] * prewittM_x[(k + 1) * 3 + (l + 1)];
					sum_y += depthData[offset_el][0] * prewittM_y[(k + 1) * 3 + (l + 1)];
				}
			}
			// the magnitude.
			int magnitude = std::sqrt(std::pow(std::abs(sum_x), 2) + std::pow(std::abs(sum_y), 2));
			uchar magn_trunchiata = (uchar)std::min(magnitude, 255);
			depthProcessedData[offset] = cv::Vec4b(magnitude, magnitude, magnitude, depthData[offset][3]);
		}
	}
}


void Filter::computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height)
{
	// glm vec 3 are the vectors from the glm library.
	// obtained with the openCV.
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			offset = y * width + x;
			// the neighbors of the current pixel (x, y)
			p_left = pointCloudData[offset - 1];
			p_right = pointCloudData[offset + 1];
			p_up = pointCloudData[offset - width];
			p_down = pointCloudData[offset + width];

			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);
			// p1 p2 p2
			// p4 p5 p6
			// p7 p8 p9
			// vec_horiz = p2 - p1 s.a.s.m.d
			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;
			// produsul vectorial.
			normal = glm::cross(vec_horiz, vec_vert);
			// to avoid division by zero !
			if (glm::length(normal) > 0.00001)
				normal = glm::normalize(normal);
			normalMeasureComputedData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}


void Filter::transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height)
{

	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			normalImageComputedData[offset] = cv::Vec4b((-(normalMeasureComputedData[offset][2]) + 1) / 2 * 255,
				(-(normalMeasureComputedData[offset][1]) + 1) / 2 * 255,
				(-(normalMeasureComputedData[offset][0]) + 1) / 2 * 255, 0);

			offset++;
		}
	}
}

void Filter::computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* myNormalMeasureData, int width, int height) {
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;
	int offset;
	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {
			offset = y * width + x;
			p_left = pointCloudData[offset - 2];
			p_right = pointCloudData[offset + 2];
			p_up = pointCloudData[offset - width * 2];
			p_down = pointCloudData[offset + width * 2];

			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);

			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;

			/*
			------>   and same for vertical direction. 5 vectors
			------>	  	
			------>
			------>
			------>
			*/
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.000001) {
				normal = glm::normalize(normal);
			}
			myNormalMeasureData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}

}

void Filter::filterNormalSobel(cv::Vec4f* normalMeasure, cv::Vec4b* normalProcessedData, int width, int height) {
	int offset;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			// I've computed the neighbors in a 3x3 vicinity.
			cv::Vec4f N1 = normalMeasure[(y - 1) * width + (x - 1)];
			cv::Vec4f N2 = normalMeasure[(y - 1) * width + (x)];
			cv::Vec4f N3 = normalMeasure[(y - 1) * width + (x + 1)];
			cv::Vec4f N4 = normalMeasure[(y)*width + (x - 1)];
			cv::Vec4f N5 = normalMeasure[(y)*width + (x)];       // center
			cv::Vec4f N6 = normalMeasure[(y)*width + (x + 1)];
			cv::Vec4f N7 = normalMeasure[(y + 1) * width + (x - 1)];
			cv::Vec4f N8 = normalMeasure[(y + 1) * width + (x)];
			cv::Vec4f N9 = normalMeasure[(y + 1) * width + (x + 1)];

			// dot product between 2 normals -> the product of each component
			auto dot = [](const cv::Vec4f& a, const cv::Vec4f& b) {
				return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
				};

			// Sobel cost following the formula:
			float Gx = (1 - dot(N1, N3)) + 2 * (1 - dot(N4, N6)) + (1 - dot(N7, N9));
			float Gy = (1 - dot(N1, N7)) + 2 * (1 - dot(N2, N8)) + (1 - dot(N3, N9));

			float G = std::sqrt(std::pow(std::abs(Gx), 2) + std::pow(std::abs(Gy), 2));
			// I will scale the value of G to fit in the range [0, 255]. [0, 1] -> [0, 255], and saturate_cast will take care of the overflow.
			uchar intensity = cv::saturate_cast<uchar>(G * 255.0f);
			offset = y * width + x;
			normalProcessedData[offset] = cv::Vec4b(intensity, intensity, intensity, 255); // the grayscale image.
		}
	}
}

// TASK 3 - segmentation using BFS.
void Filter::segmentBFSFromSeeds(const uchar* grayscaleData, const cv::Vec4f* pointCloudData, const cv::Vec4f* normalMeasureData,
	int width, int height, int numSeeds, float w1, float w2, float w3, float costThreshold, std::vector<int>& labelsOut,
	std::vector<RegionSummary>& regionsOut) {

	// w1 -> weight for intensity
	// w2 -> weight for depth
	// w3 -> weight for normal

	// costThreshold -> maximum cost to add a pixel to a region
	// labelsOut -> output labels for each pixel
	// regionsOut -> output region statistics

	// N will be the total number of pixels.
	const int N = width * height;

	// I will asign -1 to all pixels, meaning they are not assigned to any region.
	labelsOut.assign(N, -1);

	// Clear previous regions and reserve space for new regions.
	regionsOut.clear();
	regionsOut.reserve(numSeeds);

	// Step 1: Normalize intensity and depth values
	// minimum and maximum depth values in the point cloud.
	float dmin = std::numeric_limits<float>::infinity();
	float dmax = -std::numeric_limits<float>::infinity();


	for (int i = 0; i < N; ++i) {
		float z = pointCloudData[i][2];
		// I will only consider finite depth
		// values for normalization.

		// z is the depth value.
		if (std::isfinite(z)) {
			dmin = std::min(dmin, z);
			dmax = std::max(dmax, z);
		}
	}
	// If dmin and dmax are not finite or dmin >= dmax,
	// I will set them to default values.
	if (!(std::isfinite(dmin) && std::isfinite(dmax)) || dmin >= dmax) {
		dmin = 0.f; dmax = 1.f;
	}
	// Normalization functions for intensity and depth.
	auto normI = [](float I) {
		return I / 255.f;
	};
	auto normD = [=](float D) {
		float t = (D - dmin) / (dmax - dmin);
		if (!std::isfinite(t)) {
			t = 0.f;
		}
		// basically, a restriction to [0, 1].
		return clampf(t, 0.f, 1.f);
		};

	// STEP 2:
	// Randomly select seed points within the image boundaries
	std::mt19937 rng(42);  // random number generator.

	// I generate random x and y coordinates for the seeds.
	std::uniform_int_distribution<int> distX(1, width - 2);
	std::uniform_int_distribution<int> distY(1, height - 2);


	// Directions for 8-neighbors
	const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int dy[8] = { 0, 1, 1,  1,  0, -1, -1,-1 };
	// KDIR -> number of directions: 8;
	const int KDIR = 8;

	//STEP 3:
	// BFS from each seed
	for (int s = 0; s < numSeeds; ++s) {
		// Randomly select a seed point
		int sx = distX(rng);
		int sy = distY(rng);
		// linear offset in the 1D array.
		int off = sy * width + sx;
		// I'll skip if this pixel is already labeled.
		if (labelsOut[off] != -1) {
			continue;
		}

		// I'll initialize a new region with the seed pixel's stats
		RegionSummary R;
		// initial intensity value
		R.I_avg = normI(grayscaleData[off]);
		// initial depth value
		R.D_avg = normD(pointCloudData[off][2]);
		// initial normal value
		glm::vec3 Np(normalMeasureData[off][0], normalMeasureData[off][1], normalMeasureData[off][2]);
		
		// I will normalize the normal vector,
		// but if its length is too small, I will set it to (0,0,1)
		if (glm::length(Np) > 1e-6f) {
			Np = glm::normalize(Np);
		} else {
			Np = glm::vec3(0, 0, 1);
		}

		R.N_avg = Np;
		R.count = 1;
		
		// Add the new region to the list and get its ID
		int r_id = (int)regionsOut.size();
		regionsOut.push_back(R);
		labelsOut[off] = r_id;

		// The queue for BFS
		std::queue<int> q;
		q.push(off);

		while (!q.empty()) {
			int cur = q.front(); q.pop();
			int cx = cur % width;
			int cy = cur / width;

			for (int k = 0; k < KDIR; ++k) {
				int nx = cx + dx[k];
				int ny = cy + dy[k];
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
					continue;
				}
				// skip it if it's already labeled
				int noff = ny * width + nx;
				if (labelsOut[noff] != -1) {
					continue;
				}



				// STEP 4:
				// I'll compute the cost of adding this pixel to the region
				float I_p = normI(grayscaleData[noff]);
				float D_p = normD(pointCloudData[noff][2]);

				glm::vec3 N_p(normalMeasureData[noff][0], normalMeasureData[noff][1], normalMeasureData[noff][2]);
				if (glm::length(N_p) > 1e-6f) {
					N_p = glm::normalize(N_p);
				} else {
					N_p = glm::vec3(0, 0, 1);
				}

				// The current region stats
				// And for every change, the changes will be reflected in the regionsOut vector. -> &.
				RegionSummary& rr = regionsOut[r_id];

				// depth of the current pixel (the one we are expanding
				// from).
				float D_cur = normD(pointCloudData[cur][2]);

				// intensity cost -> vs region average.
				float cI = std::fabs(rr.I_avg - I_p);
				// depth cost -> vs current pixel.
				float cD = std::fabs(D_cur - D_p);

				// normal cost -> vs region average normal
				// Dot product between average normal and pixel normal
				float ndot = glm::dot(rr.N_avg, N_p);
				ndot = clampf(ndot, -1.f, 1.f);
				float cN = 0.5f * (1.0f - ndot); 
				// aduc în ~[0,1]

				// formula for cost
				float cost = w1 * cI
					+ w2 * cD
					+ w3 * cN;

				// if the cost is below the threshold,
				// I will add the pixel to the region
				if (cost <= costThreshold) {
					labelsOut[noff] = r_id;
					q.push(noff);

					// Update region averages using incremental averaging,
					// as it was explained in the course.
					float cnt = (float)rr.count;
					rr.I_avg = (rr.I_avg * cnt + I_p) / (cnt + 1.f);
					rr.D_avg = (rr.D_avg * cnt + D_p) / (cnt + 1.f);
					glm::vec3 Nacc = rr.N_avg * cnt + N_p;

					// L will be the length (norm) of the accumulated normal
					float L = glm::length(Nacc);
					// normalize the accumulated normal if its length is significant
					if (L > 1e-6f) {
						rr.N_avg = Nacc / L;
					}
					rr.count += 1;
				}
			}
		}
	}

	// I have finished BFS for all seeds, and now I will fill in any remaining unlabeled pixels,
	// in the closest region based on minimum cost.
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int off = y * width + x;
			// already labeled.
			if (labelsOut[off] != -1) {
				continue;
			}

			int bestRid = -1;
			float bestC = std::numeric_limits<float>::infinity();

			float I_p = normI(grayscaleData[off]);
			float D_p = normD(pointCloudData[off][2]);
			glm::vec3 N_p(normalMeasureData[off][0], normalMeasureData[off][1], normalMeasureData[off][2]);
			if (glm::length(N_p) > 1e-6f) {
				N_p = glm::normalize(N_p);
			} else {
				N_p = glm::vec3(0, 0, 1);
			}

			for (int k = 0; k < KDIR; ++k) {
				int nx = x + dx[k], ny = y + dy[k];
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
					continue;
				}
				int noff = ny * width + nx;
				int rid = labelsOut[noff];
				if (rid < 0) {
					continue;
				}
				const RegionSummary& rr = regionsOut[rid];

				float ndot = glm::dot(rr.N_avg, N_p);
				ndot = clampf(ndot, -1.f, 1.f);
				float cN = 0.5f * (1.f - ndot);
				// compute the cost, relative to this region statistics
				float c = w1 * std::fabs(rr.I_avg - I_p)
					+ w2 * std::fabs(rr.D_avg - D_p)
					+ w3 * cN;

				if (c < bestC) {
					bestC = c;
					bestRid = rid;
				}
			}

			// I assign the pixel to the best matching neighbor region,
			// means the one with the lowest cost
			if (bestRid != -1) {
				labelsOut[off] = bestRid;
			}
		}
	}
}


// Visualization: convert labels to a colored image
void Filter::labelsToColorImage(const std::vector<int>& labels, int width, int height, cv::Vec4b* outRGBAMatrix) {
	
	// function to generate a unique color for each region id
	auto colorFor = [](int id) -> cv::Vec4b {
		if (id < 0) {
			// It will be black for unlabeled pixels
			return cv::Vec4b(0, 0, 0, 255);
		}
		// Using a hash function to generate a color from the id
		uint32_t h = static_cast<uint32_t>(id) * 2654435761u;

		uchar r = (h) & 0xFF;
		uchar g = (h >> 8) & 0xFF;
		uchar b = (h >> 16) & 0xFF;
		return cv::Vec4b(b, g, r, 255);
	};

	// I'll assign colors to each pixel based on its label
	int N = width * height;
	for (int i = 0; i < N; i++) {
		outRGBAMatrix[i] = colorFor(labels[i]);
	}
}