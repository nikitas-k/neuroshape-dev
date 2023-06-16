#include <stdio.h>

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
