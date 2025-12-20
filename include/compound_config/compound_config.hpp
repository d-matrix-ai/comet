#pragma once

#include<optional>

#include <libconfig.h++>
#include <yaml-cpp/yaml.h>
#include <cassert>
#include <iostream>
#include <cstring>
namespace config
{

class CompoundConfig; // forward declaration

class CompoundConfigNode
{
 private:
  libconfig::Setting* LNode = nullptr;
  YAML::Node YNode;
  CompoundConfig* cConfig = nullptr;

 public:
  CompoundConfigNode(){}
  CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node _ynode);
  CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node _ynode, CompoundConfig* _cConfig);

  libconfig::Setting& getLNode() {return *LNode;}
  YAML::Node getYNode() const {return YNode;}

  /**
   * @brief return compound config node corresponding with `path`.
   */
  CompoundConfigNode lookup(const char *path) const;
  inline CompoundConfigNode lookup(const std::string &path) const
  { return(lookup(path.c_str())); }
  template <typename T>
  T get(const char *name) const { 
    T value;
    if (YNode) {
      try {
        value = YNode[name].as<T>();
      } catch (YAML::KeyNotFound& e) {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;
      }                                                                               
      catch (YAML::InvalidNode& e)                                                     
      {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;                                                                      
      }                                                                               
      catch (YAML::BadConversion& e)                                                  
      {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;                                                                      
      }                                                                               
      return value;
    }
    else {
      assert(false);
      throw std::logic_error("BAD YAML LOOKUP");
    }
  }
  template <typename T> //function getValue is a template function, T is a placeholder for a data type that will be specified when the function is called.
  bool getValue(const char *name, T &value) const {
    if (YNode) {
      if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
      try {
        value = YNode[name].as<T>();
      } catch (YAML::KeyNotFound& e) {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;
      }                                                                               
      catch (YAML::InvalidNode& e)                                                     
      {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;                                                                      
      }                                                                               
      catch (YAML::BadConversion& e)                                                  
      {                                                                               
        std::cerr << "ERROR: " << e.msg << ", at line: " << e.mark.line+1<< std::endl;
        std::cerr << "Cannot find " << name << " under root key: variables" << std::endl;
        throw e;                                                                      
      }                                                                               
      return true;
    }
    else {
      assert(false);
      return false;
    }
  }

  bool lookupValue(const char *name, bool &value) const;
  bool lookupValue(const char *name, int &value) const;
  bool lookupValue(const char *name, unsigned int &value) const;
  bool lookupValueLongOnly(const char *name, long long &value) const; // Only for values with an L like 123L
  bool lookupValueLongOnly(const char *name, unsigned long long &value) const; // Only for values with an L like 123L
  bool lookupValue(const char *name, long long &value) const;
  bool lookupValue(const char *name, unsigned long long &value) const;
  bool lookupValue(const char *name, double &value) const;
  bool lookupValue(const char *name, float &value) const;
  bool lookupValue(const char *name, const char *&value) const;
  bool lookupValue(const char *name, std::string &value) const;
  
  /// @brief Resolves the current YNode value to a string.
  std::string resolve() const;

  /// @brief Instantiates a key in a Map.
  bool instantiateKey(const char *name);
  /// @brief Scalar setter (template).
  template <typename T>
  bool setScalar(const T value);
  /// @brief Creates/appends to Sequence (template).
  template <typename T>
  bool push_back(const T value);

  inline bool lookupValue(const std::string &name, bool &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, int &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, unsigned int &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, long long &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name,
                          unsigned long long &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, double &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, float &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, const char *&value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, std::string &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool instantiateKey(const std::string &name)
  { return instantiateKey(name.c_str()); }

  bool exists(const char *name) const;

  inline bool exists(const std::string &name) const
  { return(exists(name.c_str())); }

  bool lookupArrayValue(const char* name, std::vector<std::string> &vectorValue) const;

  inline bool lookupArrayValue(const std::string &name, std::vector<std::string> &vectorValue) const
  { return(lookupArrayValue(name.c_str(), vectorValue));}

  bool isList() const;
  bool isArray() const;
  bool isMap() const;
  int getLength() const;

  CompoundConfigNode operator [](int idx) const;

  bool getArrayValue(std::vector<std::string> &vectorValue) const;
  // iterate through all maps and get the keys within a node
  bool getMapKeys(std::vector<std::string> &mapKeys) const;
};

class CompoundConfig
{
 private:
  bool useLConfig;
  libconfig::Config LConfig;
  YAML::Node YConfig;
  CompoundConfigNode root;
  CompoundConfigNode variableRoot;

 public:
  CompoundConfig(){assert(false);}
  CompoundConfig(const char* inputFile);
  CompoundConfig(char* inputFile) : CompoundConfig((const char*) inputFile) {}
  CompoundConfig(std::vector<std::string> inputFiles);
  CompoundConfig(std::string input, std::string format); // yaml file given as string

  ~CompoundConfig(){}

  libconfig::Config& getLConfig();
  YAML::Node& getYConfig();
  CompoundConfigNode getRoot() const;
  CompoundConfigNode getVariableRoot() const;

  bool hasLConfig() { return useLConfig;}

  std::vector<std::string> inFiles;

};

  std::uint64_t parseElementSize(std::string name);
  std::string parseName(std::string name);

} // namespace config
