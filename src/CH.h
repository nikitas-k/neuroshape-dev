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

const CEdge* CRichModel_Neigh(const CRichModel* model, int root)
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

CPoint3D CPoint3D_create()
{
    CPoint3D pt;
    pt.x = 0.0;
    pt.y = 0.0;
    pt.z = 0.0;
    return pt;
}

CPoint3D CPoint3D_createWithValues(double x, double y, double z)
{
    CPoint3D pt;
    pt.x = x;
    pt.y = y;
    pt.z = z;
    return pt;
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
    CRichModel model;
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
    // ... implementation of BuildSequenceTree ...
}

void InitContainers(ImprovedCH* obj)
{
    // ... implementation of InitContainers ...
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

ImprovedCH* ImprovedCH_create(const CRichModel* inputModel, int indexOfSourceVert)
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

int main()
{
    // ... implementation of main ...
    return 0;
}
