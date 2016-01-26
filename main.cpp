#include "npy.h"

#include <string>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;
using namespace elegant;

int main()
{
    Mat<double> ma = npy::load("/home/svenni/Dropbox/tmp/ma.npy");
    cout << ma << endl;

    npy::save("/home/svenni/Dropbox/tmp/ma_out.npy", ma);

    Mat<double> mb{{1,2,8}, {4,5,6}};
    cout << mb << endl;
    npy::save("/home/svenni/Dropbox/tmp/mb.npy", mb);
    Mat<double> mc = npy::load("/home/svenni/Dropbox/tmp/mb.npy");
    cout << mc << endl;
    return 0;
}

