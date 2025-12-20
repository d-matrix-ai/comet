#pragma once

#include <iostream>
#include <iomanip>
#include <cassert>

namespace arch
{

//
// Attribute.
//

template<class T>
class Attribute
{
 private:
  T t_;
  std::string name_;
  bool specified_;

 public:
  Attribute() : t_(), name_("NONAME"), specified_(false) {}
  
  Attribute(T t) : t_(t), name_("NONAME"), specified_(true) {}

  Attribute(T t, std::string name) : t_(t), name_(name), specified_(true) {}
  
  bool IsSpecified() const { return specified_; }
  
  T Get() const
  {
    assert(specified_);
    return t_;
  }

  friend std::ostream& operator << (std::ostream& out, const Attribute& a)
  {
    if (a.specified_)
    {
      // FIXME: names aren't initialized properly.
      // out << std::left << std::setw(12) << a.name_;
      // out << " : ";
      out << a.t_;
    }
    else
    {
      out << "-";
    }
    return out;
  }

  // Serialization
};

} // namespace model
