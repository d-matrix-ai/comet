#pragma once
#include "factory.hpp"
template<class T, class I>
struct RegisterClass
{
  RegisterClass(const std::string name)
  {
    FactoryBase<I>::getInstance().template RegisterDerivedType<T>(name);
  }
};

#define DEF_CLASS(className, interfaceName) \
  class className: public interfaceName

#define REGISTER_CLASS(className, interfaceName) \
  static RegisterClass<className, interfaceName> regIn##interfaceName__class##className(#className)

////RegisterClass<className, interfaceName> AutoRegistrar<className, interfaceName>::theStaticRegistrar(#className); 
//// T is a concrete object of Interface I, needed to avoid multiple translation units getting multiple static objects
//template <typename T, typename I>
//struct AutoRegistrar
//{
//    AutoRegistrar() {&theStaticRegistrar;}
//  
//    static RegisterClass<T, I> theStaticRegistrar;
//};
//
//#define REGISTER_CLASS(className, interfaceName) \
//  class className; \
//  AutoRegistrar<className, interfaceName>.theStaticRegistrar = RegisterClass<className, interfaceName>(#className); \
//  class className : public interfaceName 
