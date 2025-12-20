#pragma once
#include <map>
#include <functional>
#include <memory>
#include <string>
    
// templated subtype Instantiation
template<class DerivedType, class InterfaceName>
std::shared_ptr<InterfaceName> FactoryInstantiateObject() {
  return std::make_shared<DerivedType>();
}


template<class InterfaceName>
class FactoryBase
{
  public:
    static FactoryBase& getInstance() {
      static FactoryBase base;
      return base;
    }
    // no parameter constructor needed for interface types
    using CreatorFunction = std::function<std::shared_ptr<InterfaceName>()>;
    // templated register function derived functions
    template<class DerivedType>
    void RegisterDerivedType(const std::string name)
    { 
      using std::placeholders::_1;
      objmap[name] = FactoryInstantiateObject<DerivedType, InterfaceName>;
        //std::bind(&FactoryBase<InterfaceName>::instantiate<DerivedType>, getInstance(), _1);
    }

    std::shared_ptr<InterfaceName> create_object(const std::string& name) { 
      auto it = objmap.find(name);
      if (it == objmap.end()) {return nullptr;}
      else {return it->second();}
    }


    FactoryBase(const FactoryBase&) = delete;
    FactoryBase& operator=(const FactoryBase& other) = delete;
    FactoryBase(FactoryBase&&) = delete;
    FactoryBase& operator=(FactoryBase&&) = delete;
    ~FactoryBase() = default;

  private:
    std::map<std::string, std::function<std::shared_ptr<InterfaceName>()>> objmap{};
    FactoryBase() = default;
};
    //template<class InterfaceName>
    //template<class DerivedType>
    //void FactoryBase<InterfaceName>::RegisterDerivedType(const std::string name)
