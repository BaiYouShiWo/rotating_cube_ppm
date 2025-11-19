#include <stdio.h>
#include <stdint.h>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

namespace fs = std::filesystem;

#define SCREEN_WIDTH 720
#define SCREEN_HEIGHT 640

#define PI 3.14159265358979323846

class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {
        if (rows <= 0 || cols <= 0) throw std::invalid_argument("Matrix dimensions must be positive");
    }


    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    const std::vector<std::vector<double>>& data() const noexcept { return data_; }

    double& operator()(int r, int c) {
        check(r, c);
        return data_[r][c];
    }
    double operator()(int r, int c) const {
        check(r, c);
        return data_[r][c];
    }

    Matrix T() const {
        Matrix R(cols_, rows_);
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j)
                R(j, i) = data_[i][j];
        return R;
    }

    void dump(std::ostream& os = std::cout, int prec = 4) const {
        os << "Matrix " << rows_ << "x" << cols_ << ":\n";
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                os << std::setw(prec + 6) << data_[i][j];
            }
            os << '\n';
        }
    }

private:
    int rows_, cols_;
    std::vector<std::vector<double>> data_;

    void check(int r, int c) const {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_)
            throw std::out_of_range("Matrix index out of range");
    }
};

inline Matrix operator+(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("Matrix addition dimension mismatch");
    Matrix R(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            R(i, j) = A(i, j) + B(i, j);
    return R;
}

inline Matrix operator-(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("Matrix subtraction dimension mismatch");
    Matrix R(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            R(i, j) = A(i, j) - B(i, j);
    return R;
}

inline Matrix operator*(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("Matrix multiplication dimension mismatch");
    Matrix R(A.rows(), B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int k = 0; k < A.cols(); ++k) {
            double a = A(i, k);
            for (int j = 0; j < B.cols(); ++j) {
                R(i, j) += a * B(k, j);
            }
        }
    }
    return R;
}

inline Matrix operator*(double s, const Matrix& A) {
    Matrix R(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            R(i, j) = s * A(i, j);
    return R;
}
inline Matrix operator*(const Matrix& A, double s) { return s * A; }

class vec3{
public:
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}
    vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    vec3 operator+(const vec3& other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }

    vec3 operator-(const vec3& other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }

    vec3 operator*(float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    vec3 operator/(float scalar) const {
        return vec3(x / scalar, y / scalar, z / scalar);
    }

    float dot(const vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    vec3 cross(const vec3& other) const {
        return vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    vec3 normalize() const {
        float len = length();
        return vec3(x / len, y / len, z / len);
    }
};

Matrix drawLine(Matrix& image, int x0, int y0, int x1, int y1){
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if(x0 >=0 && x0 < SCREEN_WIDTH && y0 >=0 && y0 < SCREEN_HEIGHT)
            image(y0, x0) = 1.0f;

        if (x0 == x1 && y0 == y1) break;
        int err2 = 2 * err;
        if (err2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (err2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
    return image;
}

int write_to_ppm(Matrix& image, int timestep){
    std::string dir = "data";
    std::string filename(256, '\0');
    snprintf(filename.data(), filename.size(), "output_%03d.ppm", timestep);

    fs::path outPath = fs::path(dir) / filename.data();
    if (!fs::exists(outPath.parent_path())) {
        fs::create_directories(outPath.parent_path());
    }

    auto f = fopen(outPath.string().c_str(), "wb");
    
    fprintf(f,"P6\n%d %d\n255\n", SCREEN_WIDTH, SCREEN_HEIGHT); // Image dimensions
    
    for (int y = 0; y < SCREEN_HEIGHT; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            float v = image(y, x);
            int iv = static_cast<int>(v * 255.0f);
            if (iv < 0) iv = 0;
            if (iv > 255) iv = 255;
            fputc(iv, f);
            fputc(iv, f);
            fputc(iv, f);
        }
    }
    fclose(f);
    printf("PPM image created: %s\n", filename.c_str());
    return 0;
}

Matrix getRotateMatrix(float x_angle, float y_angle, float z_angle){
    Matrix Rx(4,4);
    Rx(0,0) = 1; Rx(0,1) = 0;                Rx(0,2) = 0;                 Rx(0,3) = 0;
    Rx(1,0) = 0; Rx(1,1) = cos(x_angle);     Rx(1,2) = -sin(x_angle);     Rx(1,3) = 0;
    Rx(2,0) = 0; Rx(2,1) = sin(x_angle);     Rx(2,2) = cos(x_angle);      Rx(2,3) = 0;
    Rx(3,0) = 0; Rx(3,1) = 0;                Rx(3,2) = 0;                 Rx(3,3) = 1;

    Matrix Ry(4,4);
    Ry(0,0) = cos(y_angle);   Ry(0,1) = 0; Ry(0,2) = sin(y_angle);    Ry(0,3) = 0;
    Ry(1,0) = 0;              Ry(1,1) = 1; Ry(1,2) = 0;               Ry(1,3) = 0;
    Ry(2,0) = -sin(y_angle);  Ry(2,1) = 0; Ry(2,2) = cos(y_angle);    Ry(2,3) = 0;
    Ry(3,0) = 0;              Ry(3,1) = 0; Ry(3,2) = 0;               Ry(3,3) = 1;

    Matrix Rz(4,4);
    Rz(0,0) = cos(z_angle);   Rz(0,1) = -sin(z_angle);    Rz(0,2) = 0; Rz(0,3) = 0;
    Rz(1,0) = sin(z_angle);   Rz(1,1) = cos(z_angle);     Rz(1,2) = 0; Rz(1,3) = 0;
    Rz(2,0) = 0;              Rz(2,1) = 0;                Rz(2,2) = 1; Rz(2,3) = 0;
    Rz(3,0) = 0;              Rz(3,1) = 0;                Rz(3,2) = 0; Rz(3,3) = 1;

    return Rz * Ry * Rx;
}

Matrix getTranslateMatrix(float tx, float ty, float tz){
    Matrix T(4,4);
    T(0,0) = 1; T(0,1) = 0; T(0,2) = 0; T(0,3) = tx;
    T(1,0) = 0; T(1,1) = 1; T(1,2) = 0; T(1,3) = ty;
    T(2,0) = 0; T(2,1) = 0; T(2,2) = 1; T(2,3) = tz;
    T(3,0) = 0; T(3,1) = 0; T(3,2) = 0; T(3,3) = 1;
    return T;
}

Matrix getViewMatrix(vec3 eye, vec3 center, vec3 up){
    vec3 f = (center - eye).normalize();
    vec3 s = f.cross(up).normalize();
    vec3 u = s.cross(f);

    Matrix V(4,4);
    V(0,0) = s.x; V(0,1) = s.y; V(0,2) = s.z; V(0,3) = -s.dot(eye);
    V(1,0) = u.x; V(1,1) = u.y; V(1,2) = u.z; V(1,3) = -u.dot(eye);
    V(2,0) = -f.x;V(2,1) = -f.y;V(2,2) = -f.z;V(2,3) = f.dot(eye);
    V(3,0) = 0;   V(3,1) = 0;   V(3,2) = 0;   V(3,3) = 1;
    return V;
}

Matrix getPerspectiveMatrix(float fov, float aspect, float near, float far){
    Matrix P(4,4);
    float f = 1.0f / tan(fov / 2.0f);
    P(0,0) = f / aspect; P(0,1) = 0;   P(0,2) = 0;                          P(0,3) = 0;
    P(1,0) = 0;          P(1,1) = f;   P(1,2) = 0;                          P(1,3) = 0;
    P(2,0) = 0;          P(2,1) = 0;   P(2,2) = (far + near) / (near - far); P(2,3) = (2 * far * near) / (near - far);
    P(3,0) = 0;          P(3,1) = 0;   P(3,2) = -1;                         P(3,3) = 0;
    return P;
}
int main(){
    auto start = std::chrono::steady_clock::now();
    
    Matrix Model = getRotateMatrix(0.f, 0.f, 0.f);
    Matrix View = getViewMatrix(vec3(0,0,4), vec3(0,0,0), vec3(0,1,0));
    Matrix Perspective = getPerspectiveMatrix(PI/2.f, (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT, 1.f, 20.0f);

    float cubeV[8][3] = {
    {-1,-1,-1},{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},
    {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}
    };

    int cubeE[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
    };


    
    auto MVP_standard = Perspective * View * Model;
    auto MVP = MVP_standard;
    int cubeV_screen[8][2];
    for(int t=0; t<240; t++){
        auto Rotate = getRotateMatrix(PI/31.f * t, 0.f, PI/31.f * t);
        MVP =  MVP_standard * Rotate;
        
        for(int i=0; i<8; i++){
            Matrix vertex(4,1);
            vertex(0,0) = cubeV[i][0];
            vertex(1,0) = cubeV[i][1];
            vertex(2,0) = cubeV[i][2];
            vertex(3,0) = 1.0f;

            Matrix transformed = MVP * vertex;
            float w = transformed(3,0);
            transformed(0,0) /= w;
            transformed(1,0) /= w;
            transformed(2,0) /= w;

            int x_screen = (transformed(0,0) + 1) * 0.5f * SCREEN_WIDTH;
            int y_screen = (1 - (transformed(1,0) + 1) * 0.5f) * SCREEN_HEIGHT;

            cubeV_screen[i][0] = x_screen;
            cubeV_screen[i][1] = y_screen;
        }

        Matrix image(SCREEN_HEIGHT, SCREEN_WIDTH);
        
        for(int i=0; i<12; i++){
            int x0 = cubeV_screen[cubeE[i][0]][0];
            int y0 = cubeV_screen[cubeE[i][0]][1];
            int x1 = cubeV_screen[cubeE[i][1]][0];
            int y1 = cubeV_screen[cubeE[i][1]][1];

            drawLine(image, x0, y0, x1, y1);
        }

        write_to_ppm(image, t);
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Elapsed: " << ms << " ms  (" << us << " us)\n";
    system("pause");
    return 0;

}