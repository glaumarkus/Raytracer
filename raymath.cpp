#include <numeric>

float dot_product(Vector v1, Vector v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector cross(Vector v1, Vector v2){

	x = v1.y * v2.z - v1.z * v2.y;
	y = v1.z * v2.x - v1.x * v2.z;
	z = v1.x * v2.y - v1.y * v2.x;

	return Vector(x, y, z);
}



Vector::Vector()
	: x(0.0f),
	y(1.0f),
	z(0.0f)
{}

Vector::Vector(const Vector& v)
	: x(v.x),
	y(v.y),
	z(v.z)
{}

Vector::Vector(float x, float y, float z)
	: x(x),
	y(y),
	z(z)
{}

Vector::Vector(float f)
	: x(f),
	y(f),
	z(f)
{}

Vector::~Vector(){}

Vector::length2()
{
	return square(x) + square(y) + square(z);
}

Vector::length()
{
	return std::sqrt(length2());
}

float Vector::normalize()
{
	float l = length();
	*this /= l;
	return l;
}

Vector Vector::normalized()
{
	Vector v(*this);
	v.normalize();
	return v;
}

Vector& Vector::operator =(const Vector& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	return *this;
}

Vector& Vector::operator +=(const Vector& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Vector& Vector::operator -=(const Vector& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

Vector& Vector::operator *=(float f)
{
	x *= f;
	y *= f;
	z *= f;
	return *this;
}

Vector& Vector::operator /=(float f)
{
	x /= f;
	y /= f;
	z /= f;
	return *this;
}

Vector Vector::operator -() const 
{
	return Vector(-x, -y, -z);
}

inline Vector operator +(const Vector& v1, const Vector& v2)
{
	return Vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);	
}