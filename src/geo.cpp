#include "stdafx.h"
#include <tchar.h>
#include "BaseModel.h"
#include "RichModel.h"
#include "ExactMethodForDGP.h"
#include "PreviousCH.h"
#include "ImprovedCHWithFilteringRule.h"
#include "XinWangImprovedCH.h"
#include <vector>
#include "Point3d.h"

extern "C" {
    void calculate_all_geodesic_distances(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces) {
        CRichModel model(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces);
        model.LoadModel();
        model.Preprocess();

        int num_vertices = model.GetNumOfVerts();
        for (int source = 0; source < num_vertices; ++source) {
            CExactMethodForDGP * algorithm = new CXinWangImprovedCH(model, source);
            algorithm->Execute();
            for (int dest = 0; dest < num_vertices; ++dest) {
                if (source != dest) {
                    vector<CPoint3D> resultpoints;
                    algorithm->BackTrace(dest, resultpoints);
                    double distance = 0; // Calculate the distance based on resultpoints
                    distance_matrix[source * num_vertices + dest] = distance;
                }
                else {
                    distance_matrix[source * num_vertices + dest] = 0;
                }
            }
            delete algorithm;
        }
    }
}