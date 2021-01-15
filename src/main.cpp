
#include <png++/png.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iterator>
#include <typeinfo>

void fillArrayFile(std::string file_name, std::vector<glm::vec3> &arr) {
    std::string line;
    std::vector<std::string> tokens;
    std::ifstream file (file_name);
    if (file.is_open()) {
        int p = 0;
        while ( getline (file, line) ) {
            // split line into components by " " (space)
            std::istringstream buf(line);
            std::istream_iterator<std::string> beg(buf), end;
            std::vector<std::string> tokens(beg, end);
            arr[p] = glm::vec3(::atof(tokens[0].c_str()), ::atof(tokens[1].c_str()), ::atof(tokens[2].c_str()));
            p++;
        }
        file.close();
    } else std::cout << "Unable to open file " << file_name << std::endl;
}
void multiplyArr(std::vector<glm::vec3> &arr, float f) {
    for(int i = 0; i < arr.size(); i++) {
        arr[i] *= f;
    }
}
void drawArray(std::vector<glm::vec3> &arr, std::string img_name, int width, int height) {
    png::image< png::rgb_pixel > final_image(width, height);
    int p = 0; // index
    for(int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            final_image.set_pixel(x, y, png::rgb_pixel(arr[p].x, arr[p].y, arr[p].z));
            p++;
        }
    }
    final_image.write("../"+img_name);
}
float ATrousFilterPixel(int idx, int color_index, int stepScale, int width, int height, std::vector<glm::vec3> &color, std::vector<glm::vec3> &normal, std::vector<glm::vec3> &position) {
    float kernel[25] = { 1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
                         1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                         3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
                         1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                         1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };
    int step_radius = 2;
    float c_phi = 0.2;
    float n_phi = 1.0;
    float p_phi = 1.0;
    glm::vec3 sum(0, 0, 0);
    float cum_w = 0;
    for (int h = -step_radius; h < step_radius; h++) { // guassian blur strategy
        for (int w = -step_radius; w < step_radius; w++) {
            for (int k = 0; k < 25; k++) { // kernel size
                int offset = (h * stepScale) * width + (w * stepScale); // offset*step*stepwidth
                int nidx = idx + offset;
                nidx = std::min(std::max(nidx, 0), height * width);

                float tv = color[idx][color_index] - color[nidx][color_index];
                float c_w = std::min(exp(-(glm::dot(tv, tv)) / c_phi), 1.0f);

                glm::vec3 t = normal[idx] - normal[nidx];
                float dist2 = std::max(dot(t, t) / (stepScale * stepScale), 0.0f);
                float n_w = std::min(exp(-(dist2) / n_phi), 1.0f);

                t = position[idx] - position[nidx];
                float p_w = std::min(exp(-(glm::dot(t, t)) / p_phi), 1.0f);

                float weight = c_w * n_w * p_w;
                sum += color[nidx][color_index] * weight * kernel[k]; // c_tmp * weight * kernel[i]
                cum_w += weight * kernel[k];
            }
        }
    }
    float lum = 0.2126f * sum.x + 0.7152f * sum.y + 0.0722f * sum.z;
    return lum / cum_w;
}
void ATrousFilterImage(int width, int height, std::vector<glm::vec3> &color, std::vector<glm::vec3> &normal, std::vector<glm::vec3> &position, std::vector<glm::vec3> &final) {
    int epochs = 3;
    std::vector<glm::vec3> input = color;
    for (int s = 0; s < epochs; s++) {
        int stepScale = pow(2, s);
        for(int c = 0; c < 3; c++) {
            for (int p = 0; p < width*height; p++) {
                final[p][c] += ATrousFilterPixel(p, c, stepScale, width, height, input, normal, position);
            }
        }
        input = final;
        if(s != epochs-1)multiplyArr(final, 0);
    }
}
int main() {
    cv::Mat img = cv::imread("../img.png");
    uint8_t* pixelPtr = (uint8_t*)img.data;
    int cn = img.channels();
    cv::Scalar_<uint8_t> bgrPixel;
    std::cout << img.rows << " x " << img.cols << std::endl;
    std::vector<glm::vec3> C(img.cols*img.rows);
    int p = 0; // index
    for(int i = 0; i < img.cols; i++) {
        for(int j = 0; j < img.rows; j++) {
            C[p] = glm::vec3(pixelPtr[i*img.cols*cn + j*cn + 2], pixelPtr[i*img.cols*cn + j*cn + 1], pixelPtr[i*img.cols*cn + j*cn + 0]) / 255.0f;
            p++;
        }
    }
    std::vector<glm::vec3> N(img.rows*img.cols); // N(p) = normal input at pixel p
    fillArrayFile("../normals.txt", N);
    std::vector<glm::vec3> P(img.rows*img.cols); // P(p) = position input at pixel p
    fillArrayFile("../positions.txt", P);
    std::vector<glm::vec3> F(img.rows*img.cols);
    for(int x = 0; x < F.size(); x++)
        F[x] = glm::vec3(0);
    ATrousFilterImage(img.cols, img.rows, C, N, P, F);
    multiplyArr(P, 255);
    multiplyArr(N, 255);
    multiplyArr(C, 255);
    multiplyArr(F, 255);
    drawArray(P, "position.png", img.cols, img.rows);
    drawArray(N, "normal.png", img.cols, img.rows);
    drawArray(C, "input.png", img.cols, img.rows);
    drawArray(F, "output.png", img.cols, img.rows);
}