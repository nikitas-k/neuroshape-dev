#include "ICH.h"
#include <stdbool.h>
#include <float.h>

typedef struct {
    bool fParentIsPseudoSource;
    char birthTime;
    int indexOfParent;
    int indexOfRootVertOfParent;
    int level;
    double disUptodate;
    double entryProp;
} InfoAtVertex;

typedef struct {
    char birthTime;
    int indexOfVert;
    double disUptodate;
} QuoteInfoAtVertex;

typedef struct {
    bool fComputationCompleted;
    bool fLocked;
    double totalLen;
    int nTotalCurves;
    int indexOfSourceVert;
    int nCountOfWindows;
    double nTotalMilliSeconds;
    int nMaxLenOfWindowQueue;
    double nMaxLenOfPseudoSources;
    int depthOfResultingTree;
    double NPE;
    double memory;
    double farestDis;
    const ICHModel* model;
    const char* nameOfAlgorithm;
    InfoAtVertex* m_InfoAtVertices;
} CExactMethodForDGP;

void BackTrace(CExactMethodForDGP* obj, int indexOfVert, vector<CPoint3D>* resultpoints);

CExactMethodForDGP* CExactMethodForDGP(const ICHModel* inputModel, int indexOfSourceVert) {
    CExactMethodForDGP* obj = malloc(sizeof(CExactMethodForDGP));
    obj->fComputationCompleted = false;
    obj->fLocked = false;
    obj->totalLen = 0;
    obj->nTotalCurves = 0;
    obj->indexOfSourceVert = indexOfSourceVert;
    obj->nCountOfWindows = 0;
    obj->nMaxLenOfPseudoSources = 0;
    obj->nMaxLenOfWindowQueue = 0;
    obj->depthOfResultingTree = 0;
    obj->nTotalMilliSeconds = 0;
    obj->NPE = 0;
    obj->memory = 0;
    obj->farestDis = 0;
    obj->model = *inputModel;
    obj->nameOfAlgorithm = "";
    obj->m_InfoAtVertices = malloc(inputModel->GetNumOfVerts() * sizeof(InfoAtVertex));
    obj->memory += inputModel->GetNumOfVerts() * sizeof(InfoAtVertex) / 1024 / 1024;
return obj;
}

CExactMethodForDGP* CreateCExactMethodForDGP(const ICHModel* inputModel, int indexOfSourceVert);
void DestroyCExactMethodForDGP(CExactMethodForDGP* obj);
void Execute(CExactMethodForDGP* obj);
void InitContainers(CExactMethodForDGP* obj);
void BuildSequenceTree(CExactMethodForDGP* obj);
void ClearContainers(CExactMethodForDGP* obj);
void FillExperimentalResults(CExactMethodForDGP* obj);
double GetRunTime(const CExactMethodForDGP* obj);
double GetMemoryCost(const CExactMethodForDGP* obj);
int GetWindowNum(const CExactMethodForDGP* obj);
int GetMaxLenOfQue(const CExactMethodForDGP* obj);
int GetDepthOfSequenceTree(const CExactMethodForDGP* obj);
double GetNPE(const CExactMethodForDGP* obj);
const char* GetAlgorithmName(const CExactMethodForDGP* obj);
bool HasBeenCompleted(const CExactMethodForDGP* obj);

double GetRunTime(const CExactMethodForDGP* obj)
{
    return obj->nTotalMilliSeconds;
}

double GetMemoryCost(const CExactMethodForDGP* obj)
{
    return obj->memory;
}

int GetWindowNum(const CExactMethodForDGP* obj)
{
    return obj->nCountOfWindows;
}

int GetMaxLenOfQue(const CExactMethodForDGP* obj)
{
    return obj->nMaxLenOfWindowQueue;
}

int GetDepthOfSequenceTree(const CExactMethodForDGP* obj)
{
    return obj->depthOfResultingTree;
}

double GetNPE(const CExactMethodForDGP* obj)
{
    return obj->NPE;
}

const char* GetAlgorithmName(const CExactMethodForDGP* obj)
{
    return obj->nameOfAlgorithm;
}

bool HasBeenCompleted(const CExactMethodForDGP* obj)
{
    return obj->fComputationCompleted;
}

CExactMethodForDGP* CreateCExactMethodForDGP(const ICHModel* inputModel, int indexOfSourceVert)
{
    CExactMethodForDGP* obj = (CExactMethodForDGP*)malloc(sizeof(CExactMethodForDGP));
    if (obj == NULL)
        return NULL;

    obj->fComputationCompleted = false;
    obj->fLocked = false;
    obj->totalLen = 0.0;
    obj->nTotalCurves = 0;
    obj->indexOfSourceVert = indexOfSourceVert;
    obj->nCountOfWindows = 0;
    obj->nTotalMilliSeconds = 0.0;
    obj->nMaxLenOfWindowQueue = 0;
    obj->nMaxLenOfPseudoSources = 0.0;
    obj->depthOfResultingTree = 0;
    obj->NPE = 0.0;
    obj->memory = 0.0;
    obj->farestDis = 0.0;
    obj->model = inputModel;
    obj->nameOfAlgorithm = NULL;
    obj->m_InfoAtVertices = NULL;

    return obj;
}

void DestroyCExactMethodForDGP(CExactMethodForDGP* obj)
{
    free(obj->m_InfoAtVertices);
    free(obj);
}

void Execute(struct CExactMethodForDGP* obj)
{
    if (obj->fComputationCompleted)
        return;
    if (!obj->fLocked)
    {
        obj->fLocked = 1;
        obj->nCountOfWindows = 0;
        obj->nMaxLenOfWindowQueue = 0;
        obj->depthOfResultingTree = 0;
        CExactMethodForDGP_InitContainers(obj);
        obj->nTotalMilliSeconds = clock();
        CExactMethodForDGP_BuildSequenceTree(obj);
        obj->nTotalMilliSeconds = clock() - obj->nTotalMilliSeconds;
        CExactMethodForDGP_FillExperimentalResults(obj);
        CExactMethodForDGP_ClearContainers(obj);
        obj->fComputationCompleted = 1;
        obj->fLocked = 0;
    }
}

void InitContainers(CExactMethodForDGP* obj)
{
    obj->m_InfoAtVertices = (InfoAtVertex*)malloc(model->GetNumOfEdges() * sizeof(InfoAtVertex));
    // Initialize other container(s) here if needed
}

void BuildSequenceTree(CExactMethodForDGP* obj)
{
    // Your implementation of the BuildSequenceTree function here
}

void ClearContainers(CExactMethodForDGP* obj)
{
    free(obj->m_InfoAtVertices);
    obj->m_InfoAtVertices = NULL;
    // Clear other container(s) here if needed
}

void FillExperimentalResults(CExactMethodForDGP* obj)
{
    // Your implementation of the FillExperimentalResults function here
}

void BackTrace(CExactMethodForDGP* obj, int indexOfVert, CPoint3D* resultpoints)
{
    if (obj->m_InfoAtVertices[indexOfVert].birthTime == -1)
    {
        assert(ICHModel_GetNumOfComponents(&(obj->model)) != 1 || ICHModel_Neigh(&(obj->model), indexOfVert)->size == 0);
        return;
    }
    vector_int vertexNodes;
    int index = indexOfVert;
    vector_int_push_back(&vertexNodes, index);
    while (index != obj->indexOfSourceVert)
    {
        int indexOfParent = obj->m_InfoAtVertices[index].indexOfParent;
        if (obj->m_InfoAtVertices[index].fParentIsPseudoSource)
        {
            index = indexOfParent;
        }
        else
        {
            index = obj->m_InfoAtVertices[index].indexOfRootVertOfParent;
        }
        vector_int_push_back(&vertexNodes, index);
    }

    for (int i = 0; i < (int)vertexNodes.size - 1; ++i)
    {
        int lastVert = vertexNodes.data[i];
        CPoint3D pt = ICHModel_ComputeShiftPoint(&(obj->model), lastVert);
        vector_CPoint3D_push_back(resultpoints, &pt);

        if (obj->m_InfoAtVertices[lastVert].fParentIsPseudoSource)
        {
            continue;
        }
        int parentEdgeIndex = obj->m_InfoAtVertices[lastVert].indexOfParent;
        int edgeIndex = ICHModel_Edge_indexOfReverseEdge(&(obj->model), parentEdgeIndex);
        double leftLen = ICHModel_Edge_length(&(obj->model), ICHModel_Edge_indexOfRightEdge(&(obj->model), parentEdgeIndex));
        double rightLen = ICHModel_Edge_length(&(obj->model), ICHModel_Edge_indexOfLeftEdge(&(obj->model), parentEdgeIndex));
        double xBack = ICHModel_Edge_length(&(obj->model), parentEdgeIndex) - ICHModel_Edge_xOfPlanarCoordOfOppositeVert(&(obj->model), parentEdgeIndex);
        double yBack = -ICHModel_Edge_yOfPlanarCoordOfOppositeVert(&(obj->model), parentEdgeIndex);
        double disToAngle = ICHModel_DistanceToIncidentAngle(&(obj->model), edgeIndex, xBack, yBack);

        double proportion = 1 - obj->m_InfoAtVertices[lastVert].entryProp;
        while (1)
        {
            CPoint3D pt1 = ICHModel_ComputeShiftPoint(&(obj->model), ICHModel_Edge_indexOfLeftVert(&(obj->model), edgeIndex));
            CPoint3D pt2 = ICHModel_ComputeShiftPoint(&(obj->model), ICHModel_Edge_indexOfRightVert(&(obj->model), edgeIndex));
            CPoint3D ptIntersection = ICHModel_CombineTwoNormalsTo(pt1, 1 - proportion, pt2, proportion);
            vector_CPoint3D_push_back(resultpoints, &ptIntersection);

            if (ICHModel_Edge_indexOfOppositeVert(&(obj->model), edgeIndex) == vertexNodes.data[i + 1])
                break;
            double oldProprotion = proportion;
            proportion = ICHModel_ProportionOnLeftEdgeByImage(&(obj->model), edgeIndex, xBack, yBack, oldProprotion);
            if (proportion >= -LENGTH_EPSILON_CONTROL && proportion <= 1)
            {
                proportion = max_double(proportion, 0);
                edgeIndex = ICHModel_Edge_indexOfLeftEdge(&(obj->model), edgeIndex);
                rightLen = disToAngle;
            }
            else
            {
                proportion = ICHModel_ProportionOnRightEdgeByImage(&(obj->model), edgeIndex, xBack, yBack, oldProprotion);
                proportion = max_double(proportion, 0);
                proportion = min_double(proportion, 1);
                edgeIndex = ICHModel_Edge_indexOfRightEdge(&(obj->model), edgeIndex);
                leftLen = disToAngle;
            }
            ICHModel_GetPointByRotatingAround(&(obj->model), edgeIndex, leftLen, rightLen, xBack, yBack);
            disToAngle = ICHModel_DistanceToIncidentAngle(&(obj->model), edgeIndex, xBack, yBack);
        }
    }
    CPoint3D pt = ICHModel_ComputeShiftPoint(&(obj->model), obj->indexOfSourceVert);
    vector_CPoint3D_push_back(resultpoints, &pt);
}

