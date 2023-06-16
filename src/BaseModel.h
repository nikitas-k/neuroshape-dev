#include "Point3d.h"

typedef struct {
    int verts[3];
} CFace;

typedef struct {
    char filename[256];
    vector<Point3D> verts;
    vector<CFace> faces;
    vector<Point3D> normalsToVerts;
    bool fBeLoaded;
} CBaseModel;

CBaseModel* CBaseModel_Create(char* filename) {
    CBaseModel* model = malloc(sizeof(CBaseModel));
    strcpy(model->filename, filename);
    model->fBeLoaded = false;
    return model;
}

void CBaseModel_Destroy(CBaseModel* model) {
    free(model);
}

void CBaseModel_LoadModel(CBaseModel* model) {
    // ... Implement the logic to load the model ...
}

int CBaseModel_GetNumOfVerts(const CBaseModel* model) {
    return model->verts.size();
}

int CBaseModel_GetNumOfFaces(const CBaseModel* model) {
    return model->faces.size();
}

const Point3D* CBaseModel_Vert(const CBaseModel* model, int vertIndex) {
    return &model->verts[vertIndex];
}

const Point3D* CBaseModel_Normal(const CBaseModel* model, int vertIndex) {
    return &model->normalsToVerts[vertIndex];
}

const CFace* CBaseModel_Face(const CBaseModel* model, int faceIndex) {
    return &model->faces[faceIndex];
}

bool CBaseModel_HasBeenLoad(const CBaseModel* model) {
    return model->fBeLoaded;
}