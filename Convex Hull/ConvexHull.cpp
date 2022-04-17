#include <iostream>
#include <algorithm>
#include <iterator>
#include <map>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>

using namespace std;


pair<int, int> Point(int a, int b)
{
	return make_pair(a, b);
}

map<pair<int, int>, bool> create_points(int N)
{
	map<pair<int, int>, bool> points;
	default_random_engine generator(time(0));
	uniform_int_distribution<int> distr(0, 50);

	for (int i = 0; i < N; i++)
		points[Point(distr(generator), distr(generator))] = 0;
	
	for (auto it = points.begin(); it != points.end(); ++it)
		cout << "(" << (*it).first.first << "," << (*it).first.second << ")" << " ";
	
	cout << endl;

	ofstream out("points");
	for (auto it = points.begin(); it != points.end(); ++it)
		out <<  (*it).first.first << "," << (*it).first.second << "\n";
	out.close();

	return points;
}

vector<int> find_maxmin(map<pair<int, int>, bool> points)
{
	int Xmax = 0;
	int Xmin = 1000000;
	int Ymax, Ymin;
	vector<int> indexes;
	int i = 0;
	
		for (auto it = points.begin(); it != points.end(); ++it)
		{
			if ((*it).first.first < Xmin)
			{
				Xmin = (*it).first.first;
				Ymin = (*it).first.second;
			}
			if ((*it).first.first >= Xmax)
			{
				Xmax = (*it).first.first;
				Ymax = (*it).first.second;
			}
		}
	
	indexes.push_back(Xmin);
	indexes.push_back(Ymin);
	indexes.push_back(Xmax);
	indexes.push_back(Ymax);

	return indexes;
}

int Distance(pair<int, int> pl, pair<int, int> pr, pair<int,int> p)
{
	int a = pr.second - pl.second;
	int b = pr.first - pl.first;
	return abs((a*p.first - b*p.second + pr.first*pl.second - pr.second*pl.first) / sqrt(a*a + b*b));
}

int Side(pair<int, int> pl, pair<int, int> pr, pair<int, int> p)
{
	int value = (p.second - pl.second) * (pr.first - pl.first) - (pr.second - pl.second) * (p.first - pl.first);

	if (value > 0)
		return 1;
	if (value < 0)
		return -1;
	return 0;
}

void QuickHull(map<pair<int, int>, bool> &points, pair<int, int> l_point, pair<int, int> r_point, int side)
{
	int N = points.size();
	int max_d = 0;
	int fl=0;
	pair<int, int> p;

	points.find(l_point)->second = 1;
	points.find(r_point)->second = 1;

	for (auto i : points)
	{
		int d = Distance(l_point, r_point, i.first);
		if (Side(l_point, r_point, i.first) == side && d > max_d)
		{
			max_d = d;
			p = i.first;
			fl++;
		}
	}
	if (fl != 0)
	{
		points.find(p)->second = 1;
	
		QuickHull(points, p, l_point, -Side(p, l_point, r_point));
		QuickHull(points, p, r_point, -Side(p, r_point, l_point));
	}
}

void ConvexHull(map<pair<int, int>, bool> points)
{
	int N = points.size();
	if (points.size() < 3)
	{
		cout << "It is impossible to build a convex hull!";
		return;
	}

	vector<int> m_points = find_maxmin(points);
	//top
	QuickHull(points, Point(m_points[0], m_points[1]), Point(m_points[2], m_points[3]), 1);
	//low
	QuickHull(points, Point(m_points[0], m_points[1]), Point(m_points[2], m_points[3]), -1);

	cout << "Points belonging to the convex hull:" << endl;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if ((*it).second == 1)
		{
			cout << "(" << (*it).first.first << "," << (*it).first.second << ")" << " ";
		}
	}
}

void main()
{
	map<pair<int, int>, bool> arr_points;
	arr_points = create_points(8);

	ConvexHull(arr_points);

	cin.get();
	cin.get();
}