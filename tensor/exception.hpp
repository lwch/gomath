#ifndef __TENSOR_EXCEPTION_HPP__
#define __TENSOR_EXCEPTION_HPP__

#include <exception>
#include <string>

class not_implemented_exception : public std::exception {
public:
  not_implemented_exception(const std::string &what) {
    _what = what + " not implemented";
  }
  ~not_implemented_exception() throw() {}
  virtual const char *what() const throw() { return _what.c_str(); }

protected:
  std::string _what;
};

#endif