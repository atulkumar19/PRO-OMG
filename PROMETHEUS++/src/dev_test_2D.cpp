#include "dev_test_2D.h"

DEV_TEST_2D::DEV_TEST_2D(){
    try{
        vfield_mat vf_mat;

        vf_mat.zeros(5,5);

        vf_mat.X.print("NODES: ");

        vf_mat.fill(1.0);

        vf_mat.Y.print("FILL Y: ");

        vf_mat.fill(2/0.0);

        vf_mat.Z.print("FILL Z: ");

        cout << vf_mat.X(9,10);
    }

    catch(...){
        cout << "EXCEPTION: during instantiation of DEV_TEST_2D" << endl;
    }
}
