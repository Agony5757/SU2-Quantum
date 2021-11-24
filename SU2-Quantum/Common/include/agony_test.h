#pragma once

#include "../include/CConfig.hpp"
#include "../include/linear_algebra/CSysSolve.hpp"
#include "../include/linear_algebra/CSysMatrix.hpp"
#include <tuple>
// #include <Windows.h>
#ifdef WIN32
#include <direct.h>
#else
#include <sys/stat.h> 
#endif

inline void mkdirectory(string filename) {

#ifdef WIN32
	mkdir(filename.c_str());
#else
	mkdir(filename.c_str(), S_IRWXU);
#endif

}

template<typename Ty> struct DenseMatrix;
template<typename Ty> struct DenseVector;

template<typename Ty>
struct DenseMatrix {
	int size;
	Ty* data;
	
	DenseMatrix(int size_) {
		// cout << size;
		size = size_;
		data = new Ty[size*size];

		for (int i = 0; i < size*size; ++i) data[i] = 0;
	}

	DenseMatrix(const DenseMatrix<Ty>& m) {
		size = m.size;
		data = new Ty[m.size*m.size];
		//memcpy(m.data, data, size*size * sizeof(Ty));

		for (int i = 0; i < size*size; ++i) data[i] = m.data[i];
	}

	/*DenseMatrix(DenseMatrix<Ty> &&m) {
		size = m.size;
		data = m.data;
	}*/

	Ty& get(int x, int y) {
		return data[x * size + y];
	}

	Ty& operator()(int x, int y) {
		return get(x, y);
	}

	DenseMatrix<Ty> operator*(DenseMatrix<Ty> m) {
		DenseMatrix<Ty> newm(size);

		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				for (int k = 0; k < size; ++k) {
					newm(i, j) += (*this)(i, k) * m(k, j);
				}
			}
		}
		return newm;
	}

	string to_string() {
		stringstream ss;
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				ss << get(i,j)<<" ";
			}
			ss << endl;
		}
		return ss.str();
	}

	void write_matlab_file(string filename) {
		ofstream out(filename, ios::out);
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				out << get(i, j) << " ";
			}
			out << endl;
		}
	}
	
	~DenseMatrix() {
		delete[] data;
	}
};

template<typename Ty>
struct DenseVector {
	int size;
	Ty* data;

	DenseVector() {
		size = 0;
		data = new Ty[0];
	}

	DenseVector(int size_) {
		size = size_;
		data = new Ty[size];

		for (int i = 0; i < size; ++i) data[i] = 0;
	}

	DenseVector(const DenseVector<Ty>& m) {
		size = m.size;
		data = new Ty[m.size];
		//memcpy(m.data, data, size * sizeof(Ty));

		for (int i = 0; i < size; ++i) data[i] = m.data[i];
	}

	/*DenseVector(DenseVector<Ty>&& v) {
		size = v.size;
		data = v.data;
	}*/

	DenseVector<Ty>& operator=(const DenseVector<Ty>& m) {
		delete[] data;
		size = m.size;
		data = new Ty[m.size];
		//memcpy(m.data, data, size * sizeof(Ty));

		for (int i = 0; i < size; ++i) data[i] = m.data[i];
		return *this;
	}

	DenseVector<Ty> operator-(const DenseVector<Ty>& v) {
		DenseVector<Ty> vout(*this);

		for (int i = 0; i < size; ++i)
			vout.data[i] -= v.data[i];
		return vout;
	}

	Ty& get(int x) {
		return data[x];
	}

	Ty& operator()(int x) {
		return get(x);
	}

	Ty& operator[](int x) {
		return get(x);
	}	

	Ty norm2() {
		Ty sum = 0;
		for (int i = 0; i < size; ++i) sum += (data[i] * data[i]);
		return sqrt(sum);
	}

	Ty norm2(int s, int n) {
		Ty sum = 0;
		for (int i = s; i < size; i += n) sum += (data[i] * data[i]);
		return sqrt(sum);
	}

	Ty normInf() {
		Ty ninf = 0;
		for (int i = 0; i < size; ++i) {
			if (abs(data[i]) > ninf) {
				ninf = abs(data[i]);
			}
		}
		return ninf;
	}

	string to_string() {
		stringstream ss;
		for (int i = 0; i < size; ++i) {
			ss << data[i] << endl;
		}
		return ss.str();
	}

	void write_matlab_file(string filename) {
		ofstream out(filename, ios::out);
		for (int i = 0; i < size - 1; ++i) {
			out << get(i) << ";";
		}
		out << get(size - 1);
	}

	int max(Ty& maxvalue) {
		int idx = 0;
		maxvalue = get(0);
		for (int i = 1; i < size; ++i) {
			if (get(i) > maxvalue) {
				maxvalue = get(i);
				idx = i;
			}
		}
		return idx;
	}

	int maxabs(Ty& maxvalue) {
		int idx = 0;
		maxvalue = abs(get(0));
		for (int i = 1; i < size; ++i) {
			if (abs(get(i)) > maxvalue) {
				maxvalue = abs(get(i));
				idx = i;
			}
		}
		return idx;
	}

	vector<double> square() {
		vector<double> values;
		values.resize(size);

		for (int i = 0; i < size; ++i) {
			values[i] = data[i] * data[i];
		}
		return values;
	}

	vector<int> get_sgn() {
		vector<int> sgns;
		sgns.resize(size);
		for (int i = 0; i < size; ++i) {
			sgns[i] = data[i] > 0 ? 1 : -1;
		}
		return sgns;
	}

	~DenseVector() {
		delete[] data;
	}
};

template<typename Ty>
DenseVector<Ty> operator*(DenseMatrix<Ty> m, DenseVector<Ty> v) {
	if (m.size != v.size) throw runtime_error("Bad size");
	DenseVector<Ty> v2(m.size);

	for (int i = 0; i < m.size; ++i) {
		for (int j = 0; j < m.size; ++j) {
			v2(i) += m(i, j) * v(j);
		}
	}
	return v2;
}

template<typename Ty>
DenseVector<Ty> operator*(DenseVector<Ty> v, Ty a) {
	DenseVector<Ty> v2(v.size);

	for (int i = 0; i < v.size; ++i) {
		v2(i) = v(i)*a;
	}
	return v2;
}

template<typename Ty>
DenseVector<Ty> operator/(DenseVector<Ty> v, Ty a) {
	return v * Ty(1.0 / a);
}

template<typename Ty>
DenseVector<Ty> get_column(DenseMatrix<Ty> A, int col) {
	DenseVector<Ty> v = DenseVector<Ty>(A.size);
	for (int i = 0; i < A.size; ++i) {
		v(i) = A(i, col);
	}
	return v;
}

template<typename Ty>
DenseMatrix<Ty> CSysMatrix2DenseMat(CSysMatrix<Ty>& A, CGeometry* geometry, int nVar, CConfig* config) {
	// get size
	auto point_num = geometry->GetnPoint();
	auto block_size = nVar * nVar;
	int size = point_num * nVar;
		
	DenseMatrix<Ty> m(size);
	auto row_ptr = A.row_ptr;
	auto col_ind = A.col_ind;
	auto matrix = A.matrix;

	for (auto row_i = 0u; row_i < point_num; row_i++) {
		for (auto index = row_ptr[row_i]; index < row_ptr[row_i + 1]; index++) {
			auto mat_begin = (index*nVar*nVar);
			for (auto iVar = 0; iVar < nVar; iVar++) {
				for (auto jVar = 0; jVar < nVar; jVar++) {
					auto x = col_ind[index] * nVar + jVar;
					auto y = row_i * nVar + iVar;
					m(y, x) = matrix[(unsigned long)(mat_begin + iVar * nVar + jVar)];
				}
			}
		}
	}
	return m;
}

template<typename Ty>
DenseVector<Ty> CSysVector2DenseVec(CSysVector<Ty>& b, CGeometry* geometry, int nVar, CConfig* config) {
	// get size
	auto point_num = geometry->GetnPoint();
	auto block_size = nVar;
	int size = point_num * block_size;

	DenseVector<Ty> v(size);

	for (auto i = 0u; i < point_num * nVar; ++i) {
		v(i) = b[i];
	}
	return v;
}

template<typename Ty>
CSysVector<Ty> DenseVec2CSysVector(DenseVector<Ty> v, int nVar) {
	int full_size = v.size;
	int size = v.size / nVar;
	return CSysVector<Ty>(size, size, nVar, (su2double*)v.data);
}

template<typename Ty>
bool operator==(DenseVector<Ty> v1, DenseVector<Ty> v2) {
	Ty tolerance = 1e-3;
	if (v1.size != v2.size) throw runtime_error("Bad size");
	if ((v1 - v2).norm2() < tolerance) return true;
	return false;
}

template<typename Ty>
void check_vec(DenseVector<Ty> v1, DenseVector<Ty> v2) {
	Ty tolerance = 1e-4;
	if (v1.size != v2.size) throw runtime_error("Bad size");
	for (int i = 0; i < v1.size; ++i) {
		if (abs(v1(i) - v2(i)) > tolerance) {
			cout << "Bad: " << i << "\t" << "v1: " << v1(i) << "\tv2:" << v2(i) << endl;
		}
	}
}

template<typename Ty>
void check_vec(DenseVector<Ty> v1, DenseVector<Ty> v2, Ty tolerance) {
	if (v1.size != v2.size) throw runtime_error("Bad size");
	for (int i = 0; i < v1.size; ++i) {
		if (abs(v1(i) - v2(i)) > tolerance) {
			stringstream ss;
			ss << "Bad: " << i << "\t" << "v1: " << v1(i) << "\tv2:" << v2(i) << endl;
			throw runtime_error(ss.str());
		}
	}
}

template<typename Ty>
void swap_two_rows(DenseMatrix<Ty>& A, DenseVector<Ty>& b, int row1, int row2) {
	swap(b(row1), b(row2));

	for (int i = 0; i < A.size; ++i) {
		swap(A(row1, i), A(row2, i));
	}
}

template<typename Ty> 
void row_elimination(DenseMatrix<Ty>& A, DenseVector<Ty>& b, int row, int row2) {
	int size = A.size;
	if (A(row, row) == 0 || A.size != b.size) {
		throw runtime_error("Bad Gaussian Solver.");
	}
	if (A(row2, row) == 0) return;
	Ty coef = A(row2, row) / A(row, row);
	for (int j = row; j < size; ++j) {
		A(row2, j) -= (A(row, j)*coef);
	}
	b(row2) -= (b(row)*coef);	
}

template<typename Ty> 
void column_elimination(DenseMatrix<Ty>& A, DenseVector<Ty> &b, DenseVector<Ty> &x) {
	int size = A.size;
	x(size - 1) = b(size - 1) / A(size - 1, size - 1);
	for (int i = size - 2; i >= 0; --i) {
		Ty b0 = b(i);
		for (int j = i + 1; j < size; ++j) {
			b0 -= (A(i, j)*x(j));
		}
		x(i) = b0 / A(i, i);
	}
}

template<typename Ty> 
DenseVector<Ty> my_linear_solver(DenseMatrix<Ty>& A, DenseVector<Ty> &b) {
	if (A.size != b.size) throw runtime_error("Bad size");
	int n = A.size;
	DenseVector<Ty> x(n);

	for (int i = 0; i < n; ++i) {
		// find maximum row
		Ty maximum = 0.0;
		int j_ = i;
		for (int j = i + 1; j < n; ++j) {
			if (abs(A(j, i)) > maximum) {
				j_ = j;
				maximum = abs(A(j, i));
			}
		}
		if (A(j_, i) == 0) {
			DenseVector<Ty> vi = get_column(A, i);
			cout << vi.norm2();
			throw runtime_error("No solution.");
		}
		// swap_two_rows(A, b, j_, i);

		for (int j = i + 1; j < n; ++j) {
			row_elimination(A, b, i, j);
		}
	}
	//cout << A.to_string();
	//cout << b.to_string();
	column_elimination(A, b, x);
	return x;
}

template<typename Ty> 
DenseMatrix<Ty> randmat(int size) {
	srand(time(0));
	DenseMatrix<Ty> m(size);
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			m(i, j) = rand()*1.0 / RAND_MAX;
		}
	}
	return m;
}

template<typename Ty> 
DenseVector<Ty> randvec(int size) {
	srand(time(0) + 999);
	DenseVector<Ty> v(size);
	for (int i = 0; i < size; ++i) {
		v(i) = rand()*1.0 / RAND_MAX;
	}
	return v;
}

template<typename Ty>
DenseVector<Ty> pick_threshold(DenseVector<Ty> &v, Ty threshold, int& counter) {
	DenseVector<Ty> v2(v.size);
	counter = 0;

	for (int i = 0; i < v.size; ++i) {
		double elem = v(i);
		if (abs(elem) > threshold) {
			counter++;
			v2(i) = elem;
		}
	}
	return v2;
}

template<typename Ty>
DenseVector<Ty> l_inf_tomography(DenseVector<Ty>& v, Ty threshold, su2double sampling_constant, int &counter) {
	DenseVector<Ty> v2(v.size);
	size_t N = sampling_constant * log2(v.size) / threshold / threshold;
	counter = 0;
	srand(time(0) + 666);
	for (int i = 0; i < v.size; ++i) {
		double elem = v(i);
		double p = abs(elem) * abs(elem) * N;
		if (p > 1) {
			v2(i) = elem;
			counter += round(abs(elem) * abs(elem) * N);
		}
		if (p <= 1) {
			double randnum = rand() * 1.0 / RAND_MAX;
			if (randnum < p) {
				v2(i) = 1.0 / N;
				counter++;
			}
		}
	}
	DenseVector<Ty> verr = (v - v2);
	Ty err = verr.normInf();
	if (err > threshold) {
		char buf[500];
		sprintf(buf, "Error Tomography. Out of threshold. LinfErr=%.6e. Threshold=%.6e", err, threshold);
		SU2_MPI::Error(string(buf), __FUNCTION__);
	}
	return v2;
}

#include <random>

template<typename Ty>
DenseVector<Ty> l_inf_tomography_v2(DenseVector<Ty>& v, Ty threshold, su2double sampling_constant, int& counter) {
	DenseVector<Ty> v2(v.size);
	size_t N = sampling_constant * log2(v.size) / threshold / threshold;
	counter = 0;
	default_random_engine e(static_cast<unsigned int>(time(nullptr)));

	for (int i = 0; i < v.size; ++i) {
		double elem = v(i);
		int sgn = v(i) >= 0 ? 1 : -1;
		double p = abs(elem) * abs(elem);
		
		binomial_distribution<size_t> ud(N, p);
		size_t k = ud(e);
		v2(i) = sgn * sqrt( (k * 1.0) / N);
	}
	DenseVector<Ty> verr = (v - v2);
	Ty err = verr.normInf();
	if (err > threshold) {
		char buf[500];
		sprintf(buf, "Error Tomography. Out of threshold. LinfErr=%.6e. Threshold=%.6e", err, threshold);
		SU2_MPI::Error(string(buf), __FUNCTION__);
	}
	return v2;
}

template<typename Ty>
tuple<vector<size_t>, vector<size_t>, size_t> histogram(vector<Ty>& dist, size_t sample_num, size_t space) {
	size_t dist_size = dist.size();
	vector<size_t> small_counter(space, 0);
	vector<size_t> big_counter(space, 0);
	size_t other_counter = 0;
	for (size_t i = 0; i < dist_size; ++i) {
		Ty p = dist[i];
		Ty mean = p * sample_num;

		if (mean < 1) {
			small_counter[(size_t)floor(mean * space)]++;
		}
		else if (mean < 10) {
			big_counter[(size_t)floor((mean - 1) * space) / 9]++;
		}
		else { other_counter++; }
	}
	return { small_counter, big_counter, other_counter };
}

template<typename Ty>
vector<size_t> histogram(vector<Ty>& dist, size_t space) {
	size_t dist_size = dist.size();
	vector<size_t> counter(space + 1, 0);
	for (size_t i = 0; i < dist_size; ++i) {
		Ty p = dist[i];
		Ty logp = - log(p);
		if (logp > space) {
			counter[space]++;
		}
		else {
			counter[(int)logp]++;
		}		
	}
	return counter;
}


template<typename Ty>
vector<size_t> sampling(vector<su2double>& dist, Ty& e, size_t sample_num) {
	size_t dist_size = dist.size();
	su2double psum = accumulate(dist.begin(), dist.end(), 0.0);
	su2double p_remain = psum;
	size_t n_remain = sample_num;
	vector<size_t> sample_result;
	sample_result.resize(dist_size);

	for (size_t i = 0; i < dist_size; ++i) {
		double p = dist[i] / p_remain;
		if (p >= 1) p = 1;
		size_t k;
		if (n_remain * p > 10 && n_remain * (1 - p) > 10 && n_remain > 10000) {
			double std = n_remain * p * (1 - p);
			std = sqrt(std);
			double mean = n_remain * p;
			normal_distribution<double> d(mean, std);
			double v = d(e);
			if (v <= 0) k = 0;			
			else k = (size_t)v;	
		}
		else {
			binomial_distribution<size_t> d(n_remain, p); 
			k = d(e);
		}

		n_remain -= k;
		p_remain -= dist[i];
		sample_result[i] = k;

		if (p_remain <= 0 || n_remain <= 0) {
			break;
		}
	}
	return sample_result;
}

template<typename Ty>
DenseVector<Ty> l_inf_tomography_v3(DenseVector<Ty>& v, Ty threshold, su2double sampling_constant, int& counter) {
	int size = v.size;
	DenseVector<Ty> v2(size);
	size_t N = sampling_constant * log2(v.size) / threshold / threshold;
	counter = 0;
	default_random_engine e(static_cast<unsigned int>(time(nullptr)));
	Ty vnorm2 = v.norm2();
	
	Ty sum_p = 0;
	size_t remain_N = N;

	vector<double> dist = v.square();
	vector<int> sgns = v.get_sgn();
	vector<size_t> n = sampling(dist, e, N);

	for (int i = 0; i < size; ++i) {
		v2[i] = sqrt(n[i] * 1.0 / N) * sgns[i];
	}

	//DenseVector<Ty> verr = (v - v2);
	//Ty err = verr.normInf();
	///*if (err > threshold) {
	//	char buf[500];
	//	sprintf(buf, "Error Tomography. Out of threshold. LinfErr=%.6e. Threshold=%.6e", err, threshold);
	//	SU2_MPI::Error(string(buf), __FUNCTION__);
	//}*/

	return v2;
}

typedef DenseMatrix<double> DenseMat;
typedef DenseVector<double> DenseVec;

template<typename Ty>
void print_CSysMatrix_Sparse(CSysMatrix<Ty>* mat, string filename) {
	ofstream out(filename, ios::out);

	// get size
	auto point_num = mat->nPoint;
	auto nVar = mat->nVar;
	auto block_size = nVar * nVar;
	int size = point_num * nVar;

	DenseMatrix<Ty> m(size);
	auto row_ptr = mat->row_ptr;
	auto col_ind = mat->col_ind;
	auto matrix = mat->matrix;

	out << point_num << endl;
	out << mat->nVar << endl;
	for (auto row_i = 0u; row_i < point_num; row_i++) {
		for (auto index = row_ptr[row_i]; index < row_ptr[row_i + 1]; index++) {
			auto mat_begin = (index * nVar * nVar);
			for (auto iVar = 0; iVar < nVar; iVar++) {
				for (auto jVar = 0; jVar < nVar; jVar++) {
					auto x = col_ind[index] * nVar + jVar;
					auto y = row_i * nVar + iVar;
					auto value = matrix[(unsigned long)(mat_begin + iVar * nVar + jVar)];
					out << x << "," << y << "," << value << endl;
				}
			}
		}
	}
}