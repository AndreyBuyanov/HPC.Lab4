#include <iostream>
#include <string>
#include <chrono>

#include "Matrix.hpp"

#pragma warning(push)
#pragma warning(disable:4334)
#pragma warning(disable:4267)
#pragma warning(disable:26451)

using namespace std::chrono;
using std::cout;
using std::endl;

std::size_t pow2(
    const int p)
{
    return static_cast<std::size_t>(1 << p);
}

void Run(
    const int d1,
    const int s1,
    const int d2,
    const int s2,
    const std::uint64_t seed)
{
    cout << "Parameters: D1=" << d1 << ", S1=" << s1 << ", D2=" << d2 << ", S2=" << s2 << endl;

    Matrix m1(pow2(d1), pow2(s1));
    Matrix m2(pow2(d2), pow2(s2));

    cout << "M1 size " << m1.Rows() << "x" << m1.Cols() << endl;
    cout << "M2 size " << m2.Rows() << "x" << m2.Cols() << endl;

    auto start = high_resolution_clock::now();
    m1.Init(seed);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "M1 initialization time: " << duration.count() << " milliseconds" << endl;

    start = high_resolution_clock::now();
    m2.Init(seed + 1);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "M2 initialization time: " << duration.count() << " milliseconds" << endl;

    start = high_resolution_clock::now();
    Matrix m3 = Matrix::CalculateContrast(m1, m2);
    stop = high_resolution_clock::now();
    duration =  duration_cast<milliseconds>(stop - start);
    cout << "Run time: " << duration.count() << " milliseconds" << endl;

    cout << "M1 first element = " << int(m1.Data()[0]) << endl;
    cout << "M2 first element = " << int(m2.Data()[0]) << endl;
    cout << "M3 first element = " << int(m3.Data()[0]) << endl;
    cout << "----------------------------------------------------------------------" << endl;
}

int main (int argc, char *argv[]){
    std::uint64_t seed = 1;
    Run(12, 12, 10, 10, seed);
    Run(12, 12, 9, 9, seed + 2);
    Run(13, 13, 9, 9, seed + 4);
    cout << "Press ENTER to exit..." << endl;
    std::cin.get();
    return 0;
}

#pragma warning(pop)
