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
// BaseModel.h: interface for the CBaseModel class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_OBJMODEL_H__138204A2_0445_4BA8_B76D_6E5374975C97__INCLUDED_)
#define AFX_OBJMODEL_H__138204A2_0445_4BA8_B76D_6E5374975C97__INCLUDED_

#include "Point3d.h"
class CBaseModel  
{
public:	
	CBaseModel(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces);
public:
	struct CFace
	{
		int verts[3];
		CFace(){}
		CFace(int x, int y, int z)
		{
			verts[0] = x;
			verts[1] = y;
			verts[2] = z;
		}
		int& operator[](int index)
		{
			return verts[index];
		}
		int operator[](int index) const
		{
			return verts[index];
		} 
	};
	
protected:	
    void ReadArray(std::vector<CPoint3D>& vertices, std::vector<CFace>& faces);
    void AdjustScaleAndComputeNormalsToVerts();

    int GetNumOfVerts() const;
    int GetNumOfFaces() const;
    void LoadModel();
    
    const CPoint3D& Vert(int vertIndex) const;
    const CPoint3D& Normal(int vertIndex) const;
    const CFace& Face(int faceIndex) const;
    bool HasBeenLoad() const;
public:
	inline int GetNumOfVerts() const;
	inline int GetNumOfFaces() const;
	void LoadModel();
	
	inline const CPoint3D& Vert(int vertIndex) const;
	inline const CPoint3D& Normal(int vertIndex) const;
	inline const CFace& Face(int faceIndex) const;
	inline bool HasBeenLoad() const;
protected:
	vector<CPoint3D> m_Verts;
	vector<CFace> m_Faces;
	vector<CPoint3D> m_NormalsToVerts;
	bool m_fBeLoaded;
};

int CBaseModel::GetNumOfVerts() const
{
	return (int)m_Verts.size();
}

int CBaseModel::GetNumOfFaces() const
{
	return (int)m_Faces.size();
}

const CPoint3D& CBaseModel::Vert(int vertIndex) const
{
	return m_Verts[vertIndex];
}

const CPoint3D& CBaseModel::Normal(int vertIndex) const
{
	return m_NormalsToVerts[vertIndex];
}

const CBaseModel::CFace& CBaseModel::Face(int faceIndex) const
{
	return m_Faces[faceIndex];
}

bool CBaseModel::HasBeenLoad() const
{
	return m_fBeLoaded;
}
#endif // !defined(AFX_OBJMODEL_H__138204A2_0445_4BA8_B76D_6E5374975C97__INCLUDED_)
