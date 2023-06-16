#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define DOUBLE_EPSILON 1e-10
#define LENGTH_EPSILON_CONTROL 1e-6
#define PI 3.14159265359
#define RateOfNormalShift 5e-3
#define ToleranceOfConvexAngle 1e-3

typedef struct {
    // Define the members of CBaseModel structure
    char filename[256];
    vector<CPoint3D> verts;
    vector<CFace> faces;
    vector<CPoint3D> normalsToVerts;
    bool fBeLoaded;
} CBaseModel;

typedef struct {
    double x, y, z;
} CPoint3D;

typedef struct {
    int verts[3];
} CFace;

typedef struct {
    int indexOfLeftVert;
    int indexOfRightVert;
    int indexOfOppositeVert;
    int indexOfLeftEdge;
    int indexOfRightEdge;
    int indexOfReverseEdge;
    int indexOfFrontFace;
    double length;
    double xOfPlanarCoordOfOppositeVert;
    double yOfPlanarCoordOfOppositeVert;
} CEdge;

typedef struct {
    char filename[256];
    CPoint3D* verts;
    CFace* faces;
    CPoint3D* normalsToVerts;
    int numOfVerts;
    int numOfFaces;
    int fBeLoaded;
} CBaseModel;

typedef struct {
    CBaseModel baseModel;
    int fLocked;
    int fBePreprocessed;
    int m_nHoles;
    int m_nIsolatedVerts;
    int* flagsForCheckingConvexVerts;
    CEdge* edges;
} CRichModel;

CRichModel* CRichModel_create(char* filename)
{
    CRichModel* obj = (CRichModel*)malloc(sizeof(CRichModel));
    if (obj != NULL) {
        strcpy(obj->baseModel.filename, filename);
        obj->baseModel.verts = NULL;
        obj->baseModel.faces = NULL;
        obj->baseModel.normalsToVerts = NULL;
        obj->baseModel.numOfVerts = 0;
        obj->baseModel.numOfFaces = 0;
        obj->baseModel.fBeLoaded = 0;
        obj->fLocked = 0;
        obj->fBePreprocessed = 0;
        obj->m_nHoles = 0;
        obj->m_nIsolatedVerts = 0;
        obj->flagsForCheckingConvexVerts = NULL;
        obj->edges = NULL;
    }
    return obj;
}

void CRichModel_destroy(CRichModel* obj)
{
    free(obj);
}

int CRichModel_IncidentVertex(const CRichModel* model, int edgeIndex)
{
    return model->m_Edges[edgeIndex].indexOfOppositeVert;
}

int CRichModel_GetNumOfValidDirectedEdges(const CRichModel* model)
{
    return model->m_Faces.size() * 3;
}

int CRichModel_GetNumOfTotalUndirectedEdges(const CRichModel* model)
{
    return model->m_Edges.size() / 2;
}

int CRichModel_GetNumOfGenera(const CRichModel* model)
{
    return (int)(CRichModel_GetNumOfTotalUndirectedEdges(model) - (CRichModel_GetNumOfVerts(model) - model->m_nIsolatedVerts) - CRichModel_GetNumOfFaces(model) - CRichModel_GetNumOfHoles(model)) / 2 + 1;
}

int CRichModel_GetNumOfComponents(const CRichModel* model)
{
    return (int)(CRichModel_GetNumOfVerts(model) - model->m_nIsolatedVerts + CRichModel_GetNumOfFaces(model) + CRichModel_GetNumOfHoles(model) - CRichModel_GetNumOfTotalUndirectedEdges(model)) / 2;
}

int CRichModel_GetNumOfHoles(const CRichModel* model)
{
    return model->m_nHoles;
}

bool CRichModel_IsClosedModel(const CRichModel* model)
{
    return CRichModel_GetNumOfValidDirectedEdges(model) == CRichModel_GetNumOfEdges(model);
}

int CRichModel_GetNumOfIsolated(const CRichModel* model)
{
    return model->m_nIsolatedVerts;
}

int CRichModel_GetNumOfEdges(const CRichModel* model)
{
    return (int)model->m_Edges.size();
}

bool CRichModel_IsConvexVert(const CRichModel* model, int index)
{
    return model->m_FlagsForCheckingConvexVerts[index];
}

bool CRichModel_IsExtremeEdge(const CRichModel* model, int edgeIndex)
{
    return model->m_Edges[edgeIndex].indexOfOppositeVert == -1;
}

bool CRichModel_IsStartEdge(const CRichModel* model, int edgeIndex)
{
    return model->m_Edges[model->m_Edges[edgeIndex].indexOfReverseEdge].indexOfOppositeVert == -1;
}

const CEdge* CRichModel_Edge(const CRichModel* model, int edgeIndex)
{
    return &model->m_Edges[edgeIndex];
}

const vector<pair<int, double> >* CRichModel_Neigh(const CRichModel* model, int root)
{
    return &model->m_NeighsAndAngles[root];
}

double CRichModel_ProportionOnEdgeByImageAndPropOnLeftEdge(const CRichModel* model, int edgeIndex, double x, double y, double proportion)
{
    double x1 = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * proportion;
    double y1 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * proportion;
    return CRichModel_ProportionOnEdgeByImage(model, edgeIndex, x1, y1, x, y);
}

double CRichModel_ProportionOnEdgeByImageAndPropOnRightEdge(const CRichModel* model, int edgeIndex, double x, double y, double proportion)
{
    double x1 = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * (1 - proportion) + model->m_Edges[edgeIndex].length * proportion;
    double y1 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * (1 - proportion);
    return CRichModel_ProportionOnEdgeByImage(model, edgeIndex, x1, y1, x, y);
}

double CRichModel_ProportionOnEdgeByImage(const CRichModel* model, int edgeIndex, double x, double y)
{
    double res = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * x;
    return res / ((y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert) * model->m_Edges[edgeIndex].length);
}

double CRichModel_ProportionOnEdgeByImage(const CRichModel* model, int edgeIndex, double x1, double y1, double x2, double y2)
{
    double res = x1 * y2 - x2 * y1;
    return res / ((y2 - y1) * model->m_Edges[edgeIndex].length);
}

double CRichModel_ProportionOnLeftEdgeByImage(const CRichModel* model, int edgeIndex, double x, double y, double proportion)
{
    double xBalance = proportion * model->m_Edges[edgeIndex].length;
    double res = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * (x - xBalance);
    return xBalance * y / res;
}

double CRichModel_ProportionOnRightEdgeByImage(const CRichModel* model, int edgeIndex, double x, double y, double proportion)
{
    double part1 = model->m_Edges[edgeIndex].length * y;
    double part2 = proportion * model->m_Edges[edgeIndex].length * model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double part3 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y;
    return (part3 + proportion * part1 - part2) / (part3 + part1 - part2);
}

void CRichModel_GetPointByRotatingAround(const CRichModel* model, int edgeIndex, double leftLen, double rightLen, double* xNew, double* yNew)
{
    *xNew = ((leftLen * leftLen - rightLen * rightLen) / model->m_Edges[edgeIndex].length + model->m_Edges[edgeIndex].length) / 2.0;
    *yNew = -sqrt(max<double>(leftLen * leftLen - *xNew * *xNew, 0));
}

void CRichModel_GetPointByRotatingAroundLeftChildEdge(const CRichModel* model, int edgeIndex, double x, double y, double* xNew, double* yNew)
{
    double leftLen = sqrt(x * x + y * y);
    double detaX = x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert;
    double detaY = y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double rightLen = sqrt(detaX * detaX + detaY * detaY);
    CRichModel_GetPointByRotatingAround(model, model->m_Edges[edgeIndex].indexOfLeftEdge, leftLen, rightLen, xNew, yNew);
}

void CRichModel_GetPointByRotatingAroundRightChildEdge(const CRichModel* model, int edgeIndex, double x, double y, double* xNew, double* yNew)
{
    double detaX = x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert;
    double detaY = y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double leftLen = sqrt(detaX * detaX + detaY * detaY);
    detaX = x - model->m_Edges[edgeIndex].length;
    double rightLen = sqrt(detaX * detaX + y * y);
    CRichModel_GetPointByRotatingAround(model, model->m_Edges[edgeIndex].indexOfRightEdge, leftLen, rightLen, xNew, yNew);
}

bool CRichModel_HasBeenProcessed(const CRichModel* model)
{
    return model->fBePreprocessed;
}

int CRichModel_GetSubindexToVert(const CRichModel* model, int root, int neigh)
{
    for (int i = 0; i < (int)CRichModel_Neigh(model, root)->size(); ++i)
    {
        if (CRichModel_Edge(model, CRichModel_Neigh(model, root)->at(i).first)->indexOfRightVert == neigh)
            return i;
    }
    return -1;
}

CPoint3D CRichModel_ComputeShiftPoint(const CRichModel* model, int indexOfVert)
{
    const CPoint3D* vert = &model->Vert(indexOfVert);
    const CPoint3D* normal = &model->Normal(indexOfVert);
    double epsilon = model->RateOfNormalShift;
    return CPoint3D_Add(vert, CPoint3D_Multiply(normal, epsilon));
}

CPoint3D CRichModel_ComputeShiftPointWithEpsilon(const CRichModel* model, int indexOfVert, double epsilon)
{
    const CPoint3D* vert = &model->Vert(indexOfVert);
    const CPoint3D* normal = &model->Normal(indexOfVert);
    return CPoint3D_Add(vert, CPoint3D_Multiply(normal, epsilon));
}
