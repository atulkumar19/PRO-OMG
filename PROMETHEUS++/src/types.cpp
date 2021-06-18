// COPYRIGHT 2015-2019 LEOPOLDO CARBAJAL

/*	This file is part of PROMETHEUS++.

    PROMETHEUS++ is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    PROMETHEUS++ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "types.h"

// Constructors:
/*
oneDimensional::ionSpecies::ionSpecies()
{
	p_BC.a_new = 0;
}

twoDimensional::ionSpecies::ionSpecies()
{

}
*/

vfield_vec::vfield_vec(unsigned int N){
	X = arma::zeros(N);
	Y = arma::zeros(N);
	Z = arma::zeros(N);
}

vfield_vec vfield_vec::operator+(vfield_vec R) {
	vfield_vec T;
	T.X = X + R.X;
	T.Y = Y + R.Y;
	T.Z = Z + R.Z;
	return (T);
}

vfield_vec vfield_vec::operator+=(vfield_vec R) {
	X += R.X;
	Y += R.Y;
	Z += R.Z;
	return (*this);
}

vfield_vec vfield_vec::operator-(vfield_vec R){
	vfield_vec T;
	T.X = X - R.X;
	T.Y = Y - R.Y;
	T.Z = Z - R.Z;
	return (T);
}

vfield_vec vfield_vec::operator-=(vfield_vec R){
	X -= R.X;
	Y -= R.Y;
	Z -= R.Z;
	return (*this);
}

vfield_vec vfield_vec::operator * (double s){
	vfield_vec T;
	T.X = s*X;
	T.Y = s*Y;
	T.Z = s*Z;
	return (T);
}

vfield_vec vfield_vec::operator *= (double s){
	X *= s;
	Y *= s;
	Z *= s;
	return (*this);
}

vfield_vec operator * (double s, vfield_vec R ) {
  return(R*s);
}

vfield_vec vfield_vec::operator / (double s){
	vfield_vec T;
	T.X = X/s;
	T.Y = Y/s;
	T.Z = Z/s;
	return (T);
}

vfield_vec vfield_vec::operator / (vfield_vec R){
	vfield_vec T;
	T.X = X/R.X;
	T.Y = Y/R.Y;
	T.Z = Z/R.Z;
	return (T);
}

vfield_vec vfield_vec::operator /= (double s){
	X /= s;
	Y /= s;
	Z /= s;
	return (*this);
}

vfield_vec vfield_vec::operator /= (vfield_vec R){
	X /= R.X;
	Y /= R.Y;
	Z /= R.Z;
	return (*this);
}

void vfield_vec::ones(unsigned int N){
	X.ones(N);
	Y.ones(N);
	Z.ones(N);
}

void vfield_vec::zeros(){
	X.zeros();
	Y.zeros();
	Z.zeros();
}

void vfield_vec::zeros(unsigned int N){
	X.zeros(N);
	Y.zeros(N);
	Z.zeros(N);
}


void vfield_vec::fill(double value){
	X.fill(value);
	Y.fill(value);
	Z.fill(value);
}



vfield_mat::vfield_mat(unsigned int N, unsigned int M){
	X = arma::zeros(N,M);
	Y = arma::zeros(N,M);
	Z = arma::zeros(N,M);
}


vfield_mat vfield_mat::operator + (vfield_mat R){
	vfield_mat T;
	T.X = X + R.X;
	T.Y = Y + R.Y;
	T.Z = Z + R.Z;
	return (T);
}

vfield_mat vfield_mat::operator += (vfield_mat R){
	X += R.X;
	Y += R.Y;
	Z += R.Z;
	return (*this);
}

vfield_mat vfield_mat::operator - (vfield_mat R){
	vfield_mat T;
	T.X = X - R.X;
	T.Y = Y - R.Y;
	T.Z = Z - R.Z;
	return (T);
}

vfield_mat vfield_mat::operator -= (vfield_mat R){
	X -= R.X;
	Y -= R.Y;
	Z -= R.Z;
	return (*this);
}

vfield_mat vfield_mat::operator * (double s){
	vfield_mat T;
	T.X = s*X;
	T.Y = s*Y;
	T.Z = s*Z;
	return (T);
}

vfield_mat vfield_mat::operator *= (double s){
	X *= s;
	Y *= s;
	Z *= s;
	return (*this);
}

vfield_mat operator * (double s, vfield_mat R ) {
  return(R*s);
}

vfield_mat vfield_mat::operator / (double s){
	vfield_mat T;
	T.X = X/s;
	T.Y = Y/s;
	T.Z = Z/s;
	return (T);
}

vfield_mat vfield_mat::operator / (vfield_mat R){
	vfield_mat T;
	T.X = X/R.X;
	T.Y = Y/R.Y;
	T.Z = Z/R.Z;
	return (T);
}

vfield_mat vfield_mat::operator /= (double s){
	X /= s;
	Y /= s;
	Z /= s;
	return (*this);
}

vfield_mat vfield_mat::operator /= (vfield_mat R){
	X /= R.X;
	Y /= R.Y;
	Z /= R.Z;
	return (*this);
}

void vfield_mat::fill(double value){
	X.fill(value);
	Y.fill(value);
	Z.fill(value);
}

void vfield_mat::ones(unsigned int N, unsigned int M){
	X.ones(N,M);
	Y.ones(N,M);
	Z.ones(N,M);
}

void vfield_mat::zeros(){
	X.zeros();
	Y.zeros();
	Z.zeros();
}

void vfield_mat::zeros(unsigned int N, unsigned int M){
	X.zeros(N,M);
	Y.zeros(N,M);
	Z.zeros(N,M);
}

/*
vfield_cube vfield_cube::operator + (vfield_cube R){
	vfield_cube T;
	T.X = X + R.X;
	T.Y = Y + R.Y;
	T.Z = Z + R.Z;
	return (T);
}

vfield_cube vfield_cube::operator += (vfield_cube R){
	X += R.X;
	Y += R.Y;
	Z += R.Z;
	return (*this);
}

vfield_cube vfield_cube::operator - (vfield_cube R){
	vfield_cube T;
	T.X = X - R.X;
	T.Y = Y - R.Y;
	T.Z = Z - R.Z;
	return (T);
}

vfield_cube vfield_cube::operator -= (vfield_cube R){
	X -= R.X;
	Y -= R.Y;
	Z -= R.Z;
	return (*this);
}

vfield_cube vfield_cube::operator * (double s){
	vfield_cube T;
	T.X = s*X;
	T.Y = s*Y;
	T.Z = s*Z;
	return (T);
}

vfield_cube vfield_cube::operator *= (double s){
	X *= s;
	Y *= s;
	Z *= s;
	return (*this);
}

vfield_cube operator * (double s, vfield_cube R ) {
  return(R*s);
}

vfield_cube vfield_cube::operator / (double s){
	vfield_cube T;
	T.X = X/s;
	T.Y = Y/s;
	T.Z = Z/s;
	return (T);
}

vfield_cube vfield_cube::operator / (vfield_cube R){
	vfield_cube T;
	T.X = X/R.X;
	T.Y = Y/R.Y;
	T.Z = Z/R.Z;
	return (T);
}

vfield_cube vfield_cube::operator /= (double s){
	X /= s;
	Y /= s;
	Z /= s;
	return (*this);
}

vfield_cube vfield_cube::operator /= (vfield_cube R){
	X /= R.X;
	Y /= R.Y;
	Z /= R.Z;
	return (*this);
}

void vfield_cube::fill(double value){
	X.fill(value);
	Y.fill(value);
	Z.fill(value);
}

void vfield_cube::ones(unsigned int N, unsigned int M, unsigned int P){
	X.ones(N,M,P);
	Y.ones(N,M,P);
	Z.ones(N,M,P);
}

void vfield_cube::zeros(){
	X.zeros();
	Y.zeros();
	Z.zeros();
}

void vfield_cube::zeros(unsigned int N, unsigned int M, unsigned int P){
	X.zeros(N,M,P);
	Y.zeros(N,M,P);
	Z.zeros(N,M,P);
}
*/

void oneDimensional::fields::zeros(unsigned int N){
	E.zeros(N);
	B.zeros(N);
}

void twoDimensional::fields::zeros(unsigned int N, unsigned int M){
	E.zeros(N,M);
	B.zeros(N,M);
}
