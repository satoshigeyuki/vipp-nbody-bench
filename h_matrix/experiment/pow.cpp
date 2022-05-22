#include <cmath>
#include <iostream>

int main()
{
  for (std::size_t p = 3; p < 7; p++) {
    std::cout << p << '\t' << std::pow(10, -p) << std::endl;
  }

}
