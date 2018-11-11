#include <boost/python.hpp>

int add(int lhs, int rhs){
    return lhs + rhs;
}

void act(boost::python::dict &x1, boost::python::list &x2){
    //exec keras prediction
    int iter_times = 1000;
    int size = x2::len();
    int pred = [2][size];
    int rewards = [2][size];

    for(int i=0; i<iter_times; i++){
        std::fillna(rewards);
        
    }
}

BOOST_PYTHON_MODULE(basic){
    using namespace boost::python;
    def("add",add);
}

