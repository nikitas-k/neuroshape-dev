#ifndef GEODESICS_H
#define GEODESICS_H
#include "stdafx.h"
#include <tchar.h>
#include "BaseModel.h"
#include "RichModel.h"
#include "ExactMethodForDGP.h"
#include "PreviousCH.h"
#include "ImprovedCHWithFilteringRule.h"
#include "XinWangImprovedCH.h"

#include <vector>
#include "Point3d.h"  // Assuming that this is the header where CPoint3D and CFace are defined
#include "RichModel.h" // Assuming that this is the header where CRichModel and CXinWangImprovedCH are defined

extern "C" {
    void calculate_all_geodesic_distances(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces);
}

#endif // GEODESICS_H