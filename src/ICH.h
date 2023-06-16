#define MAX_WINDOWS 1000
#define MAX_PSEUDO_SOURCES 1000

#include <stdbool.h>
#include <float.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define DOUBLE_EPSILON 1e-10
#define LENGTH_EPSILON_CONTROL 1e-6
#define PI 3.14159265359
#define RateOfNormalShift 5e-3
#define ToleranceOfConvexAngle 1e-3

#define MAX_PSEUDO_SOURCES 1000
#define MAX_WINDOWS 1000

// BaseModel
typedef struct {
    CPoint3D* verts;
    CFace* faces;
    CPoint3D* normalsToVerts;
    int numOfVerts;
    int numOfFaces;
    int fBeLoaded;
} CBaseModel;

void CBaseModel_AdjustScaleAndComputeNormalsToVerts(CBaseModel* model) {
    if (model->m_Verts == NULL || model->m_Verts->size == 0) {
    return;
    }
    model->m_NormalsToVerts = (CPoint3D*)malloc(model->m_Verts->size * sizeof(CPoint3D));
    m_NormalsToVerts, 0, model->m_Verts->size * sizeof(CPoint3D));
    CPoint3D center = { 0, 0, 0 };
    double sumArea = 0;
    CPoint3D sumNormal = { 0, 0, 0 };
    double deta = 0;
    
    for (int i = 0; i < model->m_Faces->size; ++i) {
    	CPoint3D normal = CPoint3D_vectorCross(CPoint3D_vectorSub(CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 0))),
  		CPoint3D_vectorSub(CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 1))),
  		CPoint3D_vectorSub(CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 2))));
    	double area = CPoint3D_length(normal);
    	CPoint3D gravity3 = CPoint3D_vectorAdd(CPoint3D_Add(CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 0))), CPoint3D_vectorAdd(CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 1))), CBaseModel_Vert(*model, CFace_Vert(model->m_Faces->data[i], 2)));
    	center += area * gravity3;
    	sumArea += area;
    	sumNormal = CPoint3D_Add(sumNormal, normal);
    	deta += CPoint3D_Dot(gravity3, normal);
    	normal.x /= area;
    	normal.y /= area;
    	normal.z /= area;
    
    	for (int j = 0; j < 3; ++j) {
    		model->m_NormalsToVerts[CFace_Vert(model->m_Faces->data[i], j)] = CPoint3D_Add(model->m_NormalsToVerts[CFace_Vert(model->m_Faces->data[i], j)], normal);
    	}
    }

    center /= sumArea * 3;
    deta -= 3 * CPoint3D_vectorDot(center, sumNormal);
    
    if (deta > 0) {
    	for (int i = 0; i < model->m_Verts->size; ++i) {
    		if (fabs(model->m_NormalsToVerts[i].x) + fabs(model->m_NormalsToVerts[i].y) + fabs(model->m_NormalsToVerts[i].z) >= FLT_EPSILON) {
    			model->m_NormalsToVerts[i] = CPoint3D_Normalize(model->m_NormalsToVerts[i]);
    		}
    	}
    }
    else {
    	for (int i = 0; i < model->m_Faces->size; ++i) {
    		int temp = CFace_Vert(model->m_Faces->data[i], 0);
    		CFace_Vert(model->m_Faces->data[i], 0) = CFace_Vert(model->m_Faces->data[i], 1);
    		CFace_Vert(model->m_Faces->data[i], 1) = temp;
    	}
    	for (int i = 0; i < model->m_Verts->size; ++i) {
    		if (fabs(model->m_NormalsToVerts[i].x) + fabs(model->m_NormalsToVerts[i].y) + fabs(model->m_NormalsToVerts[i].z) >= FLT_EPSILON) {
    			double len = CPoint3D_vectorLen(model->m_NormalsToVerts[i]);
    			model->m_NormalsToVerts[i].x /= -len;
    			model->m_NormalsToVerts[i].y /= -len;
    			model->m_NormalsToVerts[i].z /= -len;
    		}
    	}
    }
    
    CPoint3D ptUp = model->m_Verts->data[0];
    CPoint3D ptDown = model->m_Verts->data[0];
    
    for (int i = 1; i < model->m_Verts->size; ++i) {
    	if (model->m_Verts->data[i].x > ptUp.x)
    		ptUp.x = model->m_Verts->data[i].x;
    	else if (model->m_Verts->data[i].x < ptDown.x)
    		ptDown.x = model->m_Verts->data[i].x;
    	if (model->m_Verts->data[i].y > ptUp.y)
    		ptUp.y = model->m_Verts->data[i].y;
    	else if (model->m_Verts->data[i].y < ptDown.y)
    		ptDown.y = model->m_Verts->data[i].y;
    	if (model->m_Verts->data[i].z > ptUp.z)
    		ptUp.z = model->m_Verts->data[i].z;
    	else if (model->m_Verts->data[i].z < ptDown.z)
    		ptDown.z = model->m_Verts->data[i].z;
    }
    
    double maxEdgeLenOfBoundingBox = -1;
    
    if (ptUp.x - ptDown.x > maxEdgeLenOfBoundingBox)
    	maxEdgeLenOfBoundingBox = ptUp.x - ptDown.x;
    
    if (ptUp.y - ptDown.y > maxEdgeLenOfBoundingBox)
    	maxEdgeLenOfBoundingBox = ptUp.y - ptDown.y;
    
    if (ptUp.z - ptDown.z > maxEdgeLenOfBoundingBox)
    	maxEdgeLenOfBoundingBox = ptUp.z - ptDown.z;
    
    double scale = 2.0 / maxEdgeLenOfBoundingBox;
    
    for (int i = 0; i < model->m_Verts->size; ++i) {
    	model->m_Verts->data[i] = CPoint3D_vectorScale(CPoint3D_vectorSub(model->m_Verts->data[i], center), scale);
}
    
//Point3D

typedef struct {
    double x, y, z;
} CPoint3D;

CPoint3D CPoint3D_Create(double x, double y, double z) {
    CPoint3D point;
    point.x = x;
    point.y = y;
    point.z = z;
    return point;
}

CPoint3D CPoint3D_Add(const CPoint3D* pt1, const CPoint3D* pt2) {
    return CPoint3D_Create(pt1->x + pt2->x, pt1->y + pt2->y, pt1->z + pt2->z);
}

CPoint3D CPoint3D_Subtract(const CPoint3D* pt1, const CPoint3D* pt2) {
    return CPoint3D_Create(pt1->x - pt2->x, pt1->y - pt2->y, pt1->z - pt2->z);
}

CPoint3D CPoint3D_Multiply(const CPoint3D* pt, double times) {
    return CPoint3D_Create(pt->x * times, pt->y * times, pt->z * times);
}

CPoint3D CPoint3D_Divide(const CPoint3D* pt, double times) {
    return CPoint3D_Create(pt->x / times, pt->y / times, pt->z / times);
}

double CPoint3D_Length(const CPoint3D* pt) {
    return sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
}

void CPoint3D_Normalize(CPoint3D* pt) {
    double len = CPoint3D_Length(pt);
    pt->x /= len;
    pt->y /= len;
    pt->z /= len;
}

CPoint3D CPoint3D_Cross(const CPoint3D* pt1, const CPoint3D* pt2, const CPoint3D* pt3) {
    double x = (pt2->y - pt1->y) * (pt3->z - pt1->z) - (pt2->z - pt1->z) * (pt3->y - pt1->y);
    double y = (pt2->z - pt1->z) * (pt3->x - pt1->x) - (pt2->x - pt1->x) * (pt3->z - pt1->z);
    double z = (pt2->x - pt1->x) * (pt3->y - pt1->y) - (pt2->y - pt1->y) * (pt3->x - pt1->x);
    return CPoint3D_Create(x, y, z);
}

double CPoint3D_Dot(const CPoint3D* pt1, const CPoint3D* pt2) {
    return pt1->x * pt2->x + pt1->y * pt2->y + pt1->z * pt2->z;
}

double CPoint3D_AngleBetween(const CPoint3D* pt1, const CPoint3D* pt2) {
    double dotProduct = CPoint3D_Dot(pt1, pt2);
    double lenProduct = CPoint3D_Length(pt1) * CPoint3D_Length(pt2);
    return acos(dotProduct / lenProduct);
}

double CPoint3D_GetTriangleArea(const CPoint3D* pt1, const CPoint3D* pt2, const CPoint3D* pt3) {
    CPoint3D v1 = CPoint3D_Subtract(pt2, pt1);
    CPoint3D v2 = CPoint3D_Subtract(pt3, pt1);
    CPoint3D crossProduct = CPoint3D_Cross(&v1, &v2, NULL);
    return 0.5 * CPoint3D_Length(&crossProduct);
}

double CPoint3D_AngleBetweenPoints(const CPoint3D* pt1, const CPoint3D* pt2, const CPoint3D* pt3) {
    CPoint3D v1 = CPoint3D_Subtract(pt1, pt2);
    CPoint3D v2 = CPoint3D_Subtract(pt3, pt2);
    return CPoint3D_AngleBetween(&v1, &v2);
}

// CFace

typedef struct {
    int verts[3];
} CFace;

CFace CFace_new() {
    CFace face;
    face.verts[0] = 0;
    face.verts[1] = 0;
    face.verts[2] = 0;
    return face;
}

CFace CFace_newWithValues(int x, int y, int z) {
    CFace face;
    face.verts[0] = x;
    face.verts[1] = y;
    face.verts[2] = z;
    return face;
}

int* CFace_getVert(CFace* face, int index) {
    return &(face->verts[index]);
}

const int* CFace_getVertConst(const CFace* face, int index) {
    return &(face->verts[index]);
}

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

// RichModel
typedef struct {
    CBaseModel baseModel;
    int fLocked;
    int fBePreprocessed;
    int m_nHoles;
    int m_nIsolatedVerts;
    int* flagsForCheckingConvexVerts;
    CEdge* edges;
} ICHModel;

ICHModel* ICHModel_create()
{
    ICHModel* obj = (ICHModel*)malloc(sizeof(ICHModel));
    if (obj != NULL) {
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

void ICHModel_destroy(ICHModel* obj)
{
    free(obj);
}

int ICHModel_IncidentVertex(const ICHModel* model, int edgeIndex)
{
    return model->m_Edges[edgeIndex].indexOfOppositeVert;
}

int ICHModel_GetNumOfValidDirectedEdges(const ICHModel* model)
{
    return model->m_Faces.size() * 3;
}

int ICHModel_GetNumOfTotalUndirectedEdges(const ICHModel* model)
{
    return model->m_Edges.size() / 2;
}

int ICHModel_GetNumOfGenera(const ICHModel* model)
{
    return (int)(ICHModel_GetNumOfTotalUndirectedEdges(model) - (ICHModel_GetNumOfVerts(model) - model->m_nIsolatedVerts) - ICHModel_GetNumOfFaces(model) - ICHModel_GetNumOfHoles(model)) / 2 + 1;
}

int ICHModel_GetNumOfComponents(const ICHModel* model)
{
    return (int)(ICHModel_GetNumOfVerts(model) - model->m_nIsolatedVerts + ICHModel_GetNumOfFaces(model) + ICHModel_GetNumOfHoles(model) - ICHModel_GetNumOfTotalUndirectedEdges(model)) / 2;
}

int ICHModel_GetNumOfHoles(const ICHModel* model)
{
    return model->m_nHoles;
}

bool ICHModel_IsClosedModel(const ICHModel* model)
{
    return ICHModel_GetNumOfValidDirectedEdges(model) == ICHModel_GetNumOfEdges(model);
}

int ICHModel_GetNumOfIsolated(const ICHModel* model)
{
    return model->m_nIsolatedVerts;
}

int ICHModel_GetNumOfEdges(const ICHModel* model)
{
    return (int)model->m_Edges.size();
}

bool ICHModel_IsConvexVert(const ICHModel* model, int index)
{
    return model->m_FlagsForCheckingConvexVerts[index];
}

bool ICHModel_IsExtremeEdge(const ICHModel* model, int edgeIndex)
{
    return model->m_Edges[edgeIndex].indexOfOppositeVert == -1;
}

bool ICHModel_IsStartEdge(const ICHModel* model, int edgeIndex)
{
    return model->m_Edges[model->m_Edges[edgeIndex].indexOfReverseEdge].indexOfOppositeVert == -1;
}

const CEdge* ICHModel_Edge(const ICHModel* model, int edgeIndex)
{
    return &model->m_Edges[edgeIndex];
}

const CEdge* ICHModel_Neigh(const ICHModel* model, int root)
{
    return &model->m_NeighsAndAngles[root];
}

double ICHModel_ProportionOnEdgeByImageAndPropOnLeftEdge(const ICHModel* model, int edgeIndex, double x, double y, double proportion)
{
    double x1 = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * proportion;
    double y1 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * proportion;
    return ICHModel_ProportionOnEdgeByImage(model, edgeIndex, x1, y1, x, y);
}

double ICHModel_ProportionOnEdgeByImageAndPropOnRightEdge(const ICHModel* model, int edgeIndex, double x, double y, double proportion)
{
    double x1 = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * (1 - proportion) + model->m_Edges[edgeIndex].length * proportion;
    double y1 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * (1 - proportion);
    return ICHModel_ProportionOnEdgeByImage(model, edgeIndex, x1, y1, x, y);
}

double ICHModel_ProportionOnEdgeByImage(const ICHModel* model, int edgeIndex, double x, double y)
{
    double res = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * x;
    return res / ((y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert) * model->m_Edges[edgeIndex].length);
}

double ICHModel_ProportionOnEdgeByImage(const ICHModel* model, int edgeIndex, double x1, double y1, double x2, double y2)
{
    double res = x1 * y2 - x2 * y1;
    return res / ((y2 - y1) * model->m_Edges[edgeIndex].length);
}

double ICHModel_ProportionOnLeftEdgeByImage(const ICHModel* model, int edgeIndex, double x, double y, double proportion)
{
    double xBalance = proportion * model->m_Edges[edgeIndex].length;
    double res = model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * (x - xBalance);
    return xBalance * y / res;
}

double ICHModel_ProportionOnRightEdgeByImage(const ICHModel* model, int edgeIndex, double x, double y, double proportion)
{
    double part1 = model->m_Edges[edgeIndex].length * y;
    double part2 = proportion * model->m_Edges[edgeIndex].length * model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double part3 = model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert * x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert * y;
    return (part3 + proportion * part1 - part2) / (part3 + part1 - part2);
}

void ICHModel_GetPointByRotatingAround(const ICHModel* model, int edgeIndex, double leftLen, double rightLen, double* xNew, double* yNew)
{
    *xNew = ((leftLen * leftLen - rightLen * rightLen) / model->m_Edges[edgeIndex].length + model->m_Edges[edgeIndex].length) / 2.0;
    *yNew = -sqrt(max<double>(leftLen * leftLen - *xNew * *xNew, 0));
}

void ICHModel_GetPointByRotatingAroundLeftChildEdge(const ICHModel* model, int edgeIndex, double x, double y, double* xNew, double* yNew)
{
    double leftLen = sqrt(x * x + y * y);
    double detaX = x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert;
    double detaY = y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double rightLen = sqrt(detaX * detaX + detaY * detaY);
    ICHModel_GetPointByRotatingAround(model, model->m_Edges[edgeIndex].indexOfLeftEdge, leftLen, rightLen, xNew, yNew);
}

void ICHModel_GetPointByRotatingAroundRightChildEdge(const ICHModel* model, int edgeIndex, double x, double y, double* xNew, double* yNew)
{
    double detaX = x - model->m_Edges[edgeIndex].xOfPlanarCoordOfOppositeVert;
    double detaY = y - model->m_Edges[edgeIndex].yOfPlanarCoordOfOppositeVert;
    double leftLen = sqrt(detaX * detaX + detaY * detaY);
    detaX = x - model->m_Edges[edgeIndex].length;
    double rightLen = sqrt(detaX * detaX + y * y);
    ICHModel_GetPointByRotatingAround(model, model->m_Edges[edgeIndex].indexOfRightEdge, leftLen, rightLen, xNew, yNew);
}

bool ICHModel_HasBeenProcessed(const ICHModel* model)
{
    return model->fBePreprocessed;
}

int ICHModel_GetSubindexToVert(const ICHModel* model, int root, int neigh)
{
    for (int i = 0; i < (int)ICHModel_Neigh(model, root)->size(); ++i)
    {
        if (ICHModel_Edge(model, ICHModel_Neigh(model, root)->at(i).first)->indexOfRightVert == neigh)
            return i;
    }
    return -1;
}

CPoint3D ICHModel_ComputeShiftPoint(const ICHModel* model, int indexOfVert)
{
    const CPoint3D* vert = &model->Vert(indexOfVert);
    const CPoint3D* normal = &model->Normal(indexOfVert);
    double epsilon = model->RateOfNormalShift;
    return CPoint3D_Add(vert, CPoint3D_Multiply(normal, epsilon));
}

CPoint3D ICHModel_ComputeShiftPointWithEpsilon(const ICHModel* model, int indexOfVert, double epsilon)
{
    const CPoint3D* vert = &model->Vert(indexOfVert);
    const CPoint3D* normal = &model->Normal(indexOfVert);
    return CPoint3D_Add(vert, CPoint3D_Multiply(normal, epsilon));
}

int CBaseModel_GetNumOfVerts(const CBaseModel* model)
{
    return model->numOfVerts;
}

int CBaseModel_GetNumOfFaces(const CBaseModel* model)
{
    return model->numOfFaces;
}

const CPoint3D* CBaseModel_Vert(const CBaseModel* model, int vertIndex)
{
    assert(vertIndex >= 0 && vertIndex < model->numOfVerts);
    return &(model->verts[vertIndex]);
}

const CPoint3D* CBaseModel_Normal(const CBaseModel* model, int vertIndex)
{
    assert(vertIndex >= 0 && vertIndex < model->numOfVerts);
    return &(model->normalsToVerts[vertIndex]);
}

const CFace* CBaseModel_Face(const CBaseModel* model, int faceIndex)
{
    assert(faceIndex >= 0 && faceIndex < model->numOfFaces);
    return &(model->faces[faceIndex]);
}

int CBaseModel_HasBeenLoad(const CBaseModel* model)
{
    return model->fBeLoaded;
}

CPoint3D CPoint3D_add(const CPoint3D* pt1, const CPoint3D* pt2)
{
    CPoint3D result;
    result.x = pt1->x + pt2->x;
    result.y = pt1->y + pt2->y;
    result.z = pt1->z + pt2->z;
    return result;
}

CPoint3D CPoint3D_subtract(const CPoint3D* pt1, const CPoint3D* pt2)
{
    CPoint3D result;
    result.x = pt1->x - pt2->x;
    result.y = pt1->y - pt2->y;
    result.z = pt1->z - pt2->z;
    return result;
}

CPoint3D CPoint3D_multiply(const CPoint3D* pt, double times)
{
    CPoint3D result;
    result.x = pt->x * times;
    result.y = pt->y * times;
    result.z = pt->z * times;
    return result;
}

CPoint3D CPoint3D_multiplyByValue(double times, const CPoint3D* pt)
{
    return CPoint3D_multiply(pt, times);
}

CPoint3D CPoint3D_multiplyComponents(const CPoint3D* pt1, const CPoint3D* pt2)
{
    CPoint3D result;
    result.x = pt1->y * pt2->z - pt1->z * pt2->y;
    result.y = pt1->z * pt2->x - pt1->x * pt2->z;
    result.z = pt1->x * pt2->y - pt1->y * pt2->x;
    return result;
}

double CPoint3D_dotProduct(const CPoint3D* pt1, const CPoint3D* pt2)
{
    return pt1->x * pt2->x + pt1->y * pt2->y + pt1->z * pt2->z;
}

double CPoint3D_length(const CPoint3D* pt)
{
    return sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
}

void CPoint3D_normalize(CPoint3D* pt)
{
    double len = CPoint3D_length(pt);
    pt->x /= len;
    pt->y /= len;
    pt->z /= len;
}

double CPoint3D_triangleArea(const CPoint3D* pt1, const CPoint3D* pt2, const CPoint3D* pt3)
{
    CPoint3D crossProduct = CPoint3D_multiplyComponents(&CPoint3D_subtract(pt2, pt1), &CPoint3D_subtract(pt3, pt2));
    return 0.5 * CPoint3D_length(&crossProduct);
}

double CPoint3D_angleBetween(const CPoint3D* pt1, const CPoint3D* pt2)
{
    double cosAngle = CPoint3D_dotProduct(pt1, pt2) / (CPoint3D_length(pt1) * CPoint3D_length(pt2));
    if (cosAngle >= 1.0) {
        cosAngle = 1.0;
    } else if (cosAngle <= -1.0) {
        cosAngle = -1.0;
    }
    return acos(cosAngle);
}

double CPoint3D_angleBetweenThree(const CPoint3D* pt1, const CPoint3D* pt2, const CPoint3D* pt3)
{
    CPoint3D u = CPoint3D_subtract(pt2, pt1);
    CPoint3D v = CPoint3D_subtract(pt3, pt2);
    double cosAngle = CPoint3D_dotProduct(&u, &v) / (CPoint3D_length(&u) * CPoint3D_length(&v));
    if (cosAngle >= 1.0) {
        cosAngle = 1.0;
    } else if (cosAngle <= -1.0) {
        cosAngle = -1.0;
    }
    return acos(cosAngle);
}

float CPoint3D_vectorDot(const float* u, const float* v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

void CPoint3D_vectorCross(const float* u, const float* v, float* n)
{
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
}

float CPoint3D_angleBetweenFloat(const float* u, const float* v)
{
    float lenU = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    float lenV = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    float dot = CPoint3D_vectorDot(u, v) / (lenU * lenV);
    if (dot < -1.0) {
        dot = -1.0;
    }
    if (dot > 1.0) {
        dot = 1.0;
    }
    return acos(dot);
}

//ICH

typedef struct {
    int birthTime;
    int indexOfVert;
} QuoteInfoAtVertex;

typedef struct {
    double disUptodate;
    int birthTimeOfParent;
    int indexOfParent;
    int fParentIsPseudoSource;
    double entryPropOfParent;
    int fIsOnLeftSubtree;
    int level;
} QuoteWindow;

typedef struct {
    double xUponUnfolding;
    double yUponUnfolding;
    int indexOfCurEdge;
    double proportions[2];
    double disToRoot;
    int level;
} Window;

typedef struct {
    int top;
    QuoteWindow windows[MAX_WINDOWS];
} PriorityQueueForWindows;

typedef struct {
    int top;
    QuoteInfoAtVertex pseudoSources[MAX_PSEUDO_SOURCES];
} PriorityQueueForPseudoSources;

typedef struct {
    ICHModel model;
    int indexOfSourceVert;
    char* nameOfAlgorithm;
    PriorityQueueForWindows queueForWindows;
    PriorityQueueForPseudoSources queueForPseudoSources;
    InfoAtAngles infoAtAngles;
    InfoAtVertices infoAtVertices;
} ImprovedCH;

void AddIntoQueueOfPseudoSources(PriorityQueueForPseudoSources* queueOfPseudoSources, QuoteInfoAtVertex* quoteOfPseudoSource)
{
    queueOfPseudoSources->pseudoSources[++(queueOfPseudoSources->top)] = *quoteOfPseudoSource;
}

void AddIntoQueueOfWindows(PriorityQueueForWindows* queueOfWindows, QuoteWindow* quoteW)
{
    quoteW->disUptodate = GetMinDisOfWindow(*(quoteW->pWindow));
    queueOfWindows->windows[++(queueOfWindows->top)] = *quoteW;
}

double GetMinDisOfWindow(const Window* w, const Model* model) {
    double projProp = w->xUponUnfolding / model->edges[w->indexOfCurEdge].length;

    if (projProp <= w->proportions[0]) {
        double detaX = w->xUponUnfolding - w->proportions[0] * model->edges[w->indexOfCurEdge].length;
        return w->disToRoot + sqrt(detaX * detaX + w->yUponUnfolding * w->yUponUnfolding);
    }

    if (projProp >= w->proportions[1]) {
        double detaX = w->xUponUnfolding - w->proportions[1] * model->edges[w->indexOfCurEdge].length;
        return w->disToRoot + sqrt(detaX * detaX + w->yUponUnfolding * w->yUponUnfolding);
    }

    return w->disToRoot - w->yUponUnfolding;
}

int UpdateTreeDepthBackWithChoice(PriorityQueueForPseudoSources* queueOfPseudoSources, PriorityQueueForWindows* queueOfWindows)
{
    while (queueOfPseudoSources->top >= 0 && queueOfPseudoSources->pseudoSources[queueOfPseudoSources->top].birthTime != queueOfPseudoSources->pseudoSources[queueOfPseudoSources->top].indexOfVert)
        queueOfPseudoSources->top--;

    while (queueOfWindows->top >= 0)
    {
        const QuoteWindow* quoteW = &(queueOfWindows->windows[queueOfWindows->top]);
        if (quoteW->pWindow->fParentIsPseudoSource)
        {
            if (quoteW->pWindow->birthTimeOfParent != quoteW->pWindow->indexOfParent)
            {
                queueOfWindows->top--;
            }
            else
                break;
        }
        else
        {
            if (quoteW->pWindow->birthTimeOfParent == quoteW->pWindow->indexOfParent)
                break;
            else if (quoteW->pWindow->fIsOnLeftSubtree == (quoteW->pWindow->entryPropOfParent < quoteW->pWindow->entryProp))
                break;
            else
            {
                queueOfWindows->top--;
            }
        }
    }

    int fFromQueueOfPseudoSources = 0;
    if (queueOfWindows->top < 0)
    {
        if (queueOfPseudoSources->top >= 0)
        {
            const QuoteInfoAtVertex* infoOfHeadElemOfPseudoSources = &(queueOfPseudoSources->pseudoSources[queueOfPseudoSources->top]);
            // Update depthOfResultingTree
            depthOfResultingTree = max(depthOfResultingTree, 
				infoOfHeadElemOfPseudoSources.level);
            fFromQueueOfPseudoSources = 1;
        }
    }
    else
    {
        if (queueOfPseudoSources->top < 0)
        {
            const Window* infoOfHeadElemOfWindows = &(queueOfWindows->windows[queueOfWindows->top].pWindow);
            // Update depthOfResultingTree
            depthOfResultingTree = max(depthOfResultingTree, 
				infoOfHeadElemOfPseudoSources.level);
            fFromQueueOfPseudoSources = 0;
        }
        else
        {
            const QuoteInfoAtVertex* headElemOfPseudoSources = &(queueOfPseudoSources->pseudoSources[queueOfPseudoSources->top]);
            const QuoteWindow* headElemOfWindows = &(queueOfWindows->windows[queueOfWindows->top]);
            if (headElemOfPseudoSources->disUptodate <= headElemOfWindows->disUptodate)
            {
                // Update depthOfResultingTree
                depthOfResultingTree = max(depthOfResultingTree,
					m_InfoAtVertices[headElemOfPseudoSources.indexOfVert].level);
                fFromQueueOfPseudoSources = 1;
            }
            else
            {
                // Update depthOfResultingTree
                depthOfResultingTree = max(depthOfResultingTree,
					headElemOfWindows.pWindow->level);
                fFromQueueOfPseudoSources = 0;
            }
        }
    }
    return fFromQueueOfPseudoSources;
}

void BuildSequenceTree(ImprovedCH* obj)
{
    ++m_InfoAtVertices[indexOfSourceVert].birthTime;
    m_InfoAtVertices[indexOfSourceVert].level = 0;
    m_InfoAtVertices[indexOfSourceVert].disUptodate = 0;
    ComputeChildrenOfSource();
    bool fFromQueueOfPseudoSources = UpdateTreeDepthBackWithChoice();

    while (!isEmpty(m_QueueForPseudoSources) || !isEmpty(m_QueueForWindows))
    {
        if (size(m_QueueForWindows) > nMaxLenOfWindowQueue)
            nMaxLenOfWindowQueue = size(m_QueueForWindows);
        if (size(m_QueueForPseudoSources) > nMaxLenOfPseudoSources)
            nMaxLenOfPseudoSources = size(m_QueueForPseudoSources);

        if (fFromQueueOfPseudoSources)
        {
            int indexOfVert = m_QueueForPseudoSources.top().indexOfVert;
            m_QueueForPseudoSources.pop();
            ComputeChildrenOfPseudoSource(indexOfVert);
        }
        else
        {
            QuoteWindow quoteW = m_QueueForWindows.top();
            m_QueueForWindows.pop();
            ComputeChildrenOfWindow(quoteW);
            delete(quoteW.pWindow);
        }

        fFromQueueOfPseudoSources = UpdateTreeDepthBackWithChoice();
    }
}

void InitContainers(ImprovedCH* obj)
{
    m_InfoAtAngles = (InfoAtAngle*)malloc(model.GetNumOfEdges() * sizeof(InfoAtAngle));
}


void ClearContainers(ImprovedCH* obj)
{
    while (!obj->queueForWindows.empty) {
        delete obj->queueForWindows.top.pWindow;
        obj->queueForWindows.pop();
    }
    
    while (!obj->queueForPseudoSources.empty) {
        obj->queueForPseudoSources.pop();
    }
}

ImprovedCH* ImprovedCH_create(const ICHModel* inputModel, int indexOfSourceVert)
{
    ImprovedCH* obj = (ImprovedCH*)malloc(sizeof(ImprovedCH));
    if (obj != NULL)
    {
        obj->model = *inputModel;
        obj->indexOfSourceVert = indexOfSourceVert;
    }
    return obj;
}

void ImprovedCH_destroy(ImprovedCH* obj)
{
    free(obj);
}
