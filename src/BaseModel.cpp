// Copyright (c) 2005-2012 Shi-Qing Xin (xinshiqing@163.com) and Guo-Jin Wang (wanggj@zju.edu.cn).
// NOTE: this is an old version. For the lastest version, please email us.
// The code is free for research purpose, but requires a token charge for commercial purpose. 
// Users are forbidden to reproduce, republish, redistribute, or resell the code without our permission.
//
// In this code, we implemented chen and han's algorithm [1990].
// Furthermore, we gave two techniques to improve the CH algorithm.
// If you have any ideas about improving the code, we are very grateful.
// My personal website: http://sites.google.com/site/xinshiqing/Home
// 
// We are debted to Prof. Han, who gave us many helpful ideas.
// We must thank Surazhsky and Kirsanov for their knowledge share.
//
// ObjModel.cpp: implementation of the CBaseModel class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "BaseModel.h"
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
CBaseModel::CBaseModel(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces) 
{
    m_Verts = vertices;
    m_Faces = faces;
    m_fBeLoaded = false;
}

void CBaseModel::LoadModel()
{
    if (m_fBeLoaded)
        return;
    m_fBeLoaded = true;
    AdjustScaleAndComputeNormalsToVerts();
}

void CBaseModel::ReadArray(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces)
{
    map<string, int> uniqueVerts;
    vector<int> vertCorrespondence;

    for(auto &v : vertices)
    {
        ostringstream outStr;
        CPoint3D temp = v;
        if (fabs(temp.x) < FLT_EPSILON)
            temp.x = 0;
        if (fabs(temp.y) < FLT_EPSILON)
            temp.y = 0;
        if (fabs(temp.z) < FLT_EPSILON)
            temp.z = 0;
        outStr << setiosflags(ios_base::fixed) << setprecision(5) << temp.x << " " << temp.y << " " << temp.z;
        map<string, int>::iterator pos = uniqueVerts.find(outStr.str());
        if (pos == uniqueVerts.end())
        {
            int oldSize = (int)uniqueVerts.size();
            uniqueVerts[outStr.str()] = oldSize;
            vertCorrespondence.push_back(oldSize);
            m_Verts.push_back(temp);
        }
        else
        {
            vertCorrespondence.push_back(pos->second);
        }
    }
    m_Verts.swap(vector<CPoint3D>(m_Verts));

    for(auto &f : faces)
    {
        m_Faces.push_back(CFace());
        CFace& lastFace = m_Faces[m_Faces.size() - 1];            
        lastFace[0] = f[0];
        lastFace[1] = f[1];
        lastFace[2] = f[2];
        for (int j = 0; j < 3; ++j)
        {
            lastFace[j] = vertCorrespondence[lastFace[j] - 1];
        }
    }
    m_Faces.swap(vector<CFace>(m_Faces));
}

void CBaseModel::ReadVertsAndFaces(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces)
{
    ReadArray(vertices, faces);
}

void CBaseModel::AdjustScaleAndComputeNormalsToVerts()
{
	if (m_Verts.empty())
		return;
	m_NormalsToVerts.resize(m_Verts.size(), CPoint3D(0, 0, 0));
	CPoint3D center(0, 0, 0);
	double sumArea(0);
	CPoint3D sumNormal(0, 0, 0);
	double deta(0);
	for (int i = 0; i < (int)m_Faces.size(); ++i)
	{
		CPoint3D normal = VectorCross(Vert(Face(i)[0]),
			Vert(Face(i)[1]),
			Vert(Face(i)[2]));
		double area = normal.Len();
		CPoint3D gravity3 = Vert(Face(i)[0]) +	Vert(Face(i)[1]) + Vert(Face(i)[2]);
		center += area * gravity3;
		sumArea += area;
		sumNormal += normal;
		deta += gravity3 ^ normal;
		normal.x /= area;
		normal.y /= area;
		normal.z /= area;
		for (int j = 0; j < 3; ++j)
		{
			m_NormalsToVerts[Face(i)[j]] += normal;
		}
	}
	center /= sumArea * 3;
	deta -= 3 * (center ^ sumNormal);
	if (deta > 0)
	{
		for (int i = 0; i < GetNumOfVerts(); ++i)
		{
			if (fabs(m_NormalsToVerts[i].x)
				+ fabs(m_NormalsToVerts[i].y)
				+ fabs(m_NormalsToVerts[i].z) >= FLT_EPSILON)
			{					
				m_NormalsToVerts[i].Normalize();
			}
		}
	}
	else
	{
		for (int i = 0; i < GetNumOfFaces(); ++i)
		{
			int temp = m_Faces[i][0];
			m_Faces[i][0] = m_Faces[i][1];
			m_Faces[i][1] = temp;
		}
		for (int i = 0; i < GetNumOfVerts(); ++i)
		{
			if (fabs(m_NormalsToVerts[i].x)
				+ fabs(m_NormalsToVerts[i].y)
				+ fabs(m_NormalsToVerts[i].z) >= FLT_EPSILON)
			{					
				double len = m_NormalsToVerts[i].Len();
				m_NormalsToVerts[i].x /= -len;
				m_NormalsToVerts[i].y /= -len;
				m_NormalsToVerts[i].z /= -len;
			}
		}
	}

	CPoint3D ptUp(m_Verts[0]);
	CPoint3D ptDown(m_Verts[0]);
	for (int i = 1; i < GetNumOfVerts(); ++i)
	{
		if (m_Verts[i].x > ptUp.x)
			ptUp.x = m_Verts[i].x;
		else if (m_Verts[i].x < ptDown.x)
			ptDown.x = m_Verts[i].x;
		if (m_Verts[i].y > ptUp.y)
			ptUp.y = m_Verts[i].y;
		else if (m_Verts[i].y < ptDown.y)
			ptDown.y = m_Verts[i].y;
		if (m_Verts[i].z > ptUp.z)
			ptUp.z = m_Verts[i].z;
		else if (m_Verts[i].z < ptDown.z)
			ptDown.z = m_Verts[i].z;
	}	

	double maxEdgeLenOfBoundingBox = -1;
	if (ptUp.x - ptDown.x > maxEdgeLenOfBoundingBox)
		maxEdgeLenOfBoundingBox = ptUp.x - ptDown.x;
	if (ptUp.y - ptDown.y > maxEdgeLenOfBoundingBox)
		maxEdgeLenOfBoundingBox = ptUp.y - ptDown.y;
	if (ptUp.z - ptDown.z > maxEdgeLenOfBoundingBox)
		maxEdgeLenOfBoundingBox = ptUp.z - ptDown.z;
	double scale = 2.0 / maxEdgeLenOfBoundingBox;
	for (int i = 0; i < GetNumOfVerts(); ++i)
	{
		m_Verts[i] -= center;
		m_Verts[i] *= scale;
	}	
}