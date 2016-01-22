#include <iostream>
#include <armadillo>
#include <boost/regex.hpp>
#include <regex>

using namespace std;

namespace elegant {
namespace npy {

class Array
{
public:
    Array() {}
};

Array load(std::string filename)
{
    ifstream file(filename);

    string magicPrefix = "\x93NUMPY";
    int magicPrefixLength = magicPrefix.size();

    vector<char> magicBuffer(magicPrefixLength);
    file.read(&magicBuffer[0], magicBuffer.size());
    string resultingString(magicBuffer.begin(), magicBuffer.end());
    if(resultingString != magicPrefix) {
        throw std::runtime_error("The magic string is not correct");
    }
    uint8_t majorVersion = 0;
    uint8_t minorVersion = 0;
    file.read(reinterpret_cast<char*>(&majorVersion), 1);
    file.read(reinterpret_cast<char*>(&minorVersion), 1);
    cout << int(majorVersion) << "." << int(minorVersion) << endl;

    size_t headerLength = 0;
    if(majorVersion == 1 && minorVersion == 0) {
        uint16_t headerLength16 = 0;
        file.read(reinterpret_cast<char*>(&headerLength16), 2);
        headerLength = headerLength16;
    } else if(majorVersion == 2 && minorVersion == 0) {
        uint32_t headerLength32 = 0;
        file.read(reinterpret_cast<char*>(&headerLength32), 4);
        headerLength = headerLength32;
    }
    cout << int(headerLength) << endl;

    vector<char> headerBuffer(headerLength);
    file.read(&headerBuffer[0], headerLength);
    string header(headerBuffer.begin(), headerBuffer.end());

    // {'descr': '<i8', 'fortran_order': False, 'shape': (8,), }

    string::const_iterator start = header.begin();
    string::const_iterator end = header.end();

    bool fortranOrder = false;
    vector<size_t> shape;

    // match parts on the form "'key': value,"
    smatch keyPairMatch;
    while(regex_search(start, end, keyPairMatch, regex("('(.*?)'\\s*?:\\s*((\\(.*?\\)|.*?)*?),)"))) {
        string key = keyPairMatch[2];
        string value = keyPairMatch[3];
        if(key == "descr") {
            smatch descrMatch;
            if(regex_search(value, descrMatch, regex("'(<|>)(.*?)'"))) {
                string endian = descrMatch[1];
                string datatype = descrMatch[2];
                if(endian == ">") {
                    throw runtime_error("Big endian not supported");
                }
            }
        }
        if(key == "fortran_order") {
            if(value == "False") {
                fortranOrder = false;
            } else {
                fortranOrder = true;
                throw runtime_error("Fortran order is not supported");
            }
        }
        if(key == "shape") {
            smatch shapeMatch;
            string::const_iterator shapeStart = value.begin();
            string::const_iterator shapeEnd = value.end();
            while(regex_search(shapeStart, shapeEnd, shapeMatch, regex("([0-9]+?)(?:\\s*(?:,|\\)))"))) {
                string shapeValue = shapeMatch[1];
                shape.push_back(stoi(shapeValue));
                shapeStart = shapeMatch[0].second;
            }
        }
        start = keyPairMatch[0].second; // move on to the next match
    }
    for(size_t shapeSize : shape) {
        cout << "Shape: " << shapeSize << endl;
    }
    return Array();
}

}
}

using namespace std;
using namespace arma;
using namespace elegant;



int main()
{
    npy::load("/home/svenni/tmp/test3.npy");
    return 0;
}

