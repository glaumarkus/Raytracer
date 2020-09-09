#ifndef RAYMATH_H
#define RAYMATH_H


inline float square(float n)
{
	return n * n;
}


struct Vector
{
	float x, y, z;

	Vector();
	Vector(const Vector& v);
	Vector(float x, float y, float z);
	Vector(float f);

	virtual ~Vector();

	inline float length();
	inline float length2();

	float normalize();
	Vector normalized();

	Vector& operator =(const Vector& v);
	Vector& operator +=(const Vector& v);
	Vector& operator -=(const Vector& v);
	Vector& operator *=(float f);
	Vector& operator /=(float f);
	Vector operator -() const;

};

#endif // RAYMATH_H