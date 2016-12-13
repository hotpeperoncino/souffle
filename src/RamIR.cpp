#include <chrono>
#include <regex>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "RamTranslator.h"
#include "RamExecutor.h"
#include "RamVisitor.h"
#include "RamAutoIndex.h"
#include "RamLogger.h"
#include "AstRelation.h"
#include "BinaryOperator.h"

#include "AstVisitor.h"
#include "RuleScheduler.h"
#include "TypeSystem.h"

namespace {

  class IndexMap {

    typedef std::map<RamRelationIdentifier, RamAutoIndex> data_t;
    typedef typename data_t::iterator iterator;

    std::map<RamRelationIdentifier, RamAutoIndex> data;

  public:

    RamAutoIndex& operator[](const RamRelationIdentifier& rel) {
      return data[rel];
    }

    const RamAutoIndex& operator[](const RamRelationIdentifier& rel) const {
      const static RamAutoIndex empty;
      auto pos = data.find(rel);
      return (pos != data.end()) ? pos->second : empty;
    }

    iterator begin() {
      return data.begin();
    }

    iterator end() {
      return data.end();
    }

  };

  std::set<RamRelationIdentifier> getReferencedRelations(const RamOperation& op) {
    std::set<RamRelationIdentifier> res;
    visitDepthFirst(op, [&](const RamNode& node) {
	if (auto scan = dynamic_cast<const RamScan*>(&node)) {
	  res.insert(scan->getRelation());
	} else if (auto agg = dynamic_cast<const RamAggregate*>(&node)) {
	  res.insert(agg->getRelation());
	} else if (auto project = dynamic_cast<const RamProject*>(&node)) {
	  res.insert(project->getRelation());
	  if (project->hasFilter()) {
	    res.insert(project->getFilter());
	  }
	} else if (auto notExist = dynamic_cast<const RamNotExists*>(&node)) {
	  res.insert(notExist->getRelation());
	}
      });
    return res;
  }


  class RamIRinst {
    enum RamNodeType type;
    int code;
    RamNode &node;
  public:
    RamIRinst(enum RamNodeType type, int code, RamNode &node) :
      type(type), code(code), node(node) {

    }
  };
  class irstream {
    std::vector<RamNode*> irs;
  public:    
    irstream& operator<<(RamNode *p) {
      irs.push_back(p);
      return *this;
    }
  };

  class IRgen : public RamVisitor<void, irstream&> {

    const RamExecutorConfig& config;

    std::function<void(irstream&,const RamNode*)> rec;

    struct IRgencode {
      IRgen& p;
      const RamNode& node;
      IRgencode(IRgen& p, const RamNode& n) : p(p), node(n) {}
      IRgencode(const IRgencode& other) : p(other.p), node(other.node) {}
      friend irstream& operator<<(irstream& out, const IRgencode& p) {
	p.p.visit(p.node, out);
	return out;
      }
    };

  public:

    IRgen(const RamExecutorConfig& config, const IndexMap&)
      : config(config) {
      rec = [&](irstream& out, const RamNode* node) {
	this->visit(*node, out);
      };
    }

    // -- relation statements --

    void visitCreate(const RamCreate& create, irstream& out) {
      out << new RamIRinst(RN_Create, create);
    }

    void visitFact(const RamFact& fact, irstream& out) {
      out << new RamIRinst(RN_Fact, fact);
    }

    void visitLoad(const RamLoad& load, irstream& out) {
      out << new RamIRinst(RN_Load, load);      
    }

    void visitStore(const RamStore& store, irstream& out) {
      out << new RamIRinst(RN_Store, store);
    }

    void visitInsert(const RamInsert& insert, irstream& out) {

      // enclose operation with a check for an empty relation
      std::set<RamRelationIdentifier> input_relations;
      visitDepthFirst(insert, [&](const RamScan& scan) {
	  input_relations.insert(scan.getRelation());
	});
      if (!input_relations.empty()) {
	out << "if (" << join(input_relations, "&&", [&](irstream& out, const RamRelationIdentifier& rel){
	    out << "!" << this->getRelationName(rel) << ".empty()";
	  }) << ") ";
      }

      // outline each search operation to improve compilation time
      out << "[&]()";

      // enclose operation in its own scope
      out << "{\n";

      // create proof counters
      if (config.isLogging()) {
	out << "std::atomic<uint64_t> num_failed_proofs(0);\n";
      }

      // check whether loop nest can be parallelized
      bool parallel = false;
      if (const RamScan* scan = dynamic_cast<const RamScan*>(&insert.getOperation())) {

	// if it is a full scan
	if (scan->getRangeQueryColumns() == 0 && !scan->isPureExistenceCheck()) {

	  // yes it can!
	  parallel = true;

	  // partition outermost relation
	  out << "auto part = " << getRelationName(scan->getRelation()) << ".partition();\n";

	  // build a parallel block around this loop nest
	  out << "PARALLEL_START;\n";

	}
      }

      // add local counters
      if (config.isLogging()) {
	out << "uint64_t private_num_failed_proofs = 0;\n";
      }

      // create operation contexts for this operation
      for(const RamRelationIdentifier& rel : getReferencedRelations(insert.getOperation())) {
	out << "CREATE_OP_CONTEXT(" << getOpContextName(rel) << ","<< getRelationName(rel) << ".createContext());\n";
      }

      out << print(insert.getOperation());

      // aggregate proof counters
      if (config.isLogging()) {
	out << "num_failed_proofs += private_num_failed_proofs;\n";
      }

      if (parallel) out << "PARALLEL_END;\n";       // end parallel

      // aggregate proof counters
      if (config.isLogging()) {

	// get target relation
	RamRelationIdentifier rel;
	visitDepthFirst(insert, [&](const RamProject& project) {
	    rel = project.getRelation();
	  });

	// build log message
	auto &clause = insert.getOrigin();
	std::string clauseText = toString(clause);
	replace(clauseText.begin(), clauseText.end(), '"', '\'');
	replace(clauseText.begin(), clauseText.end(), '\n', ' ');

	std::ostringstream line;
	line << "p-proof-counter;" << rel.getName() << ";" << clause.getSrcLoc() << ";" << clauseText << ";";
	std::string label = line.str();

	// print log entry
	out << "{ auto lease = getOutputLock().acquire(); ";
	out << "profile << R\"(#" << label << ";)\" << num_failed_proofs << \"\\n\";\n";
	out << "}";
      }

      out << "}\n";       // end lambda
      out << "();";       // call lambda
    }

    void visitMerge(const RamMerge& merge, irstream& out) {
      out << getRelationName(merge.getTargetRelation()) << ".insertAll("
	  << getRelationName(merge.getSourceRelation())
	  << ");\n";
    }

    void visitClear(const RamClear& clear, irstream& out) {
      out << getRelationName(clear.getRelation()) << ".purge();\n";
    }

    void visitDrop(const RamDrop& drop, irstream& out) {
      std::string name = getRelationName(drop.getRelation());
      bool isTemp = (name.find("rel__temp1_")==0) || (name.find("rel__temp2_")==0);
      if (!config.isDebug() || isTemp) {
	out << name << ".purge();\n";
      }
    }

    void visitPrintSize(const RamPrintSize& print, irstream& out) {
    }

    void visitLogSize(const RamLogSize& print, irstream& out) {
      out << "{ auto lease = getOutputLock().acquire(); \n";
      out << "profile << R\"(" << print.getLabel() << ")\" <<  " << getRelationName(print.getRelation()) << ".size() << \"\\n\";\n";
      out << "}";
    }

    // -- control flow statements --

    void visitSequence(const RamSequence& seq, irstream& out) {
      out << print(seq.getFirst());
      out << print(seq.getSecond());
    }

    void visitParallel(const RamParallel& parallel, irstream& out) {
      auto stmts = parallel.getStatements();

      // special handling cases
      if (stmts.empty()) return;

      // a single statement => save the overhead
      if (stmts.size() == 1) {
	out << print(stmts[0]);
	return;
      }

      // more than one => parallel sections

      // start parallel section
      out << "SECTIONS_START;\n";

      // put each thread in another section
      for(const auto& cur : stmts) {
	out << "SECTION_START;\n";
	out << print(cur);
	out << "SECTION_END\n";
      }

      // done
      out << "SECTIONS_END;\n";
    }

    void visitLoop(const RamLoop& loop, irstream& out) {
      out << "for(;;) {\n" << print(loop.getBody()) << "}\n";
    }

    void visitExit(const RamExit& exit, irstream& out) {
      out << "if(" << print(exit.getCondition()) << ") break;\n";
    }

    void visitLogTimer(const RamLogTimer& timer, irstream& out) {
      // create local scope for name resolution
      out << "{\n";

      // create local timer
      out << "\tRamLogger logger(R\"(" << timer.getLabel() << ")\",profile);\n";

      // insert statement to be measured
      visit(timer.getNested(), out);

      // done
      out << "}\n";
    }

    // -- operations --

    void visitSearch(const RamSearch& search, irstream& out) {
      auto condition = search.getCondition();
      if(condition) {
	out << "if( " << print(condition) << ") {\n"
	    << print(search.getNestedOperation())
	    << "}\n";
	if (config.isLogging()) {
	  out << " else { ++private_num_failed_proofs; }";
	}
      } else {
	out << print(search.getNestedOperation());
      }
    }

    void visitScan(const RamScan& scan, irstream& out) {

      // get relation name
      const auto& rel = scan.getRelation();
      auto relName = getRelationName(rel);
      auto ctxName = "READ_OP_CONTEXT(" + getOpContextName(rel) + ")";
      auto level = scan.getLevel();

      // if this search is a full scan
      if (scan.getRangeQueryColumns() == 0) {
	if (scan.isPureExistenceCheck()) {
	  out << "if(!" << relName << ".empty()) {\n";
	} else if (scan.getLevel() == 0) {
	  // make this loop parallel
	  out << "pfor(auto it = part.begin(); it<part.end(); ++it) \n";
	  out << "for(const auto& env0 : *it) {\n";
	} else {
	  out << "for(const auto& env" << level << " : " << relName << ") {\n";
	}
	visitSearch(scan, out);
	out << "}\n";
	return;
      }

      // check list of keys
      auto arity = rel.getArity();
      const auto& rangePattern = scan.getRangePattern();

      // a lambda for printing boundary key values
      auto printKeyTuple = [&]() {
	for(size_t i=0; i<arity; i++) {
	  if (rangePattern[i] != nullptr) {
	    out << this->print(rangePattern[i]);
	  } else {
	    out << "0";
	  }
	  if (i+1 < arity) {
	    out << ",";
	  }
	}
      };

      // get index to be queried
      auto keys = scan.getRangeQueryColumns();
      auto index = toIndex(keys);

      // if it is a equality-range query
      out << "const Tuple<RamDomain," << arity << "> key({"; printKeyTuple(); out << "});\n";
      out << "auto range = " << relName << ".equalRange" << index << "(key," << ctxName << ");\n";
      if (config.isLogging()) {
	out << "if (range.empty()) ++private_num_failed_proofs;\n";
      }
      if (scan.isPureExistenceCheck()) {
	out << "if(!range.empty()) {\n";
      } else {
	out << "for(const auto& env" << level << " : range) {\n";
      }
      visitSearch(scan, out);
      out << "}\n";
      return;
    }

    void visitLookup(const RamLookup& lookup, irstream& out) {
      auto arity = lookup.getArity();

      // get the tuple type working with
      std::string tuple_type = "ram::Tuple<RamDomain," + toString(arity) + ">";

      // look up reference
      out << "auto ref = env" << lookup.getReferenceLevel() << "[" << lookup.getReferencePosition() << "];\n";
      out << "if (isNull<" << tuple_type << ">(ref)) continue;\n";
      out << tuple_type << " env" << lookup.getLevel() << " = unpack<" << tuple_type << ">(ref);\n";

      out << "{\n";

      // continue with condition checks and nested body
      visitSearch(lookup, out);

      out << "}\n";
    }

    void visitAggregate(const RamAggregate& aggregate, irstream& out) {

      // get some properties
      const auto& rel = aggregate.getRelation();
      auto arity = rel.getArity();
      auto relName = getRelationName(rel);
      auto ctxName = "READ_OP_CONTEXT(" + getOpContextName(rel) + ")";
      auto level = aggregate.getLevel();


      // get the tuple type working with
      std::string tuple_type = "ram::Tuple<RamDomain," + toString(arity) + ">";

      // declare environment variable
      out << tuple_type << " env" << level << ";\n";

      // special case: counting of number elements in a full relation
      if (aggregate.getFunction() == RamAggregate::COUNT && aggregate.getRangeQueryColumns() == 0) {
	// shortcut: use relation size
	out << "env" << level << "[0] = " << relName << ".size();\n";
	visitSearch(aggregate, out);
	return;
      }

      // init result
      std::string init;
      switch(aggregate.getFunction()){
      case RamAggregate::MIN: init = "MAX_RAM_DOMAIN"; break;
      case RamAggregate::MAX: init = "MIN_RAM_DOMAIN"; break;
      case RamAggregate::COUNT: init = "0"; break;
      }
      out << "RamDomain res = " << init << ";\n";

      // get range to aggregate
      auto keys = aggregate.getRangeQueryColumns();

      // check whether there is an index to use
      if (keys == 0) {

	// no index => use full relation
	out << "auto& range = " << relName << ";\n";

      } else {

	// a lambda for printing boundary key values
	auto printKeyTuple = [&]() {
	  for(size_t i=0; i<arity; i++) {
	    if (aggregate.getPattern()[i] != nullptr) {
	      out << this->print(aggregate.getPattern()[i]);
	    } else {
	      out << "0";
	    }
	    if (i+1 < arity) {
	      out << ",";
	    }
	  }
	};

	// get index
	auto index = toIndex(keys);
	out << "const " << tuple_type << " key({"; printKeyTuple();  out << "});\n";
	out << "auto range = " << relName << ".equalRange" << index << "(key," << ctxName << ");\n";

      }

      // add existence check
      if(aggregate.getFunction() != RamAggregate::COUNT) {
	out << "if(!range.empty()) {\n";
      }

      // aggregate result
      out << "for(const auto& cur : range) {\n";

      // create aggregation code
      if (aggregate.getFunction() == RamAggregate::COUNT) {

	// count is easy
	out << "++res\n;";

      } else {

	// pick function
	std::string fun = "min";
	switch(aggregate.getFunction()) {
	case RamAggregate::MIN: fun = "std::min"; break;
	case RamAggregate::MAX: fun = "std::max"; break;
	case RamAggregate::COUNT: assert(false);
	}

	out << "env" << level << " = cur;\n";
	out << "res = " << fun << "(res,"; visit(*aggregate.getTargetExpression(),out); out << ");\n";
      }

      // end aggregator loop
      out << "}\n";

      // write result into environment tuple
      out << "env" << level << "[0] = res;\n";

      // continue with condition checks and nested body
      out << "{\n";

      auto condition = aggregate.getCondition();
      if(condition) {
	out << "if( " << print(condition) << ") {\n";
	visitSearch(aggregate, out);
	out  << "}\n";
	if (config.isLogging()) {
	  out << " else { ++private_num_failed_proofs; }";
	}
      } else {
	visitSearch(aggregate, out);
      }

      out << "}\n";

      // end conditional nested block
      if(aggregate.getFunction() != RamAggregate::COUNT) {
	out << "}\n";
      }
    }

    void visitProject(const RamProject& project, irstream& out) {
      const auto& rel = project.getRelation();
      auto arity = rel.getArity();
      auto relName = getRelationName(rel);
      auto ctxName = "READ_OP_CONTEXT(" + getOpContextName(rel) + ")";

      // check condition
      auto condition = project.getCondition();
      if (condition) {
	out << "if (" << print(condition) << ") {\n";
      }

      // create projected tuple
      out << "Tuple<RamDomain," << arity << "> tuple({"
	  << join(project.getValues(), ",", rec)
	  << "});\n";

      // check filter
      if (project.hasFilter()) {
	auto relFilter = getRelationName(project.getFilter());
	auto ctxFilter = "READ_OP_CONTEXT(" + getOpContextName(project.getFilter()) + ")";
	out << "if (!" << relFilter << ".contains(tuple," << ctxFilter << ")) {";
      }

      // insert tuple
      if (config.isLogging()) {
	out << "if (!(" << relName << ".insert(tuple," << ctxName << "))) { ++private_num_failed_proofs; }\n";
      } else {
	out << relName << ".insert(tuple," << ctxName << ");\n";
      }

      // end filter
      if (project.hasFilter()) {
	out << "}";

	// add fail counter
	if (config.isLogging()) {
	  out << " else { ++private_num_failed_proofs; }";
	}
      }

      // end condition
      if (condition) {
	out << "}\n";

	// add fail counter
	if (config.isLogging()) {
	  out << " else { ++private_num_failed_proofs; }";
	}
      }


    }


    // -- conditions --

    void visitAnd(const RamAnd& c, irstream& out) {
      out << "((" << print(c.getLHS()) << ") && (" << print(c.getRHS()) << "))";
    }

    void visitBinaryRelation(const RamBinaryRelation& rel, irstream& out) {
      switch (rel.getOperator()) {

	// comparison operators
      case BinaryRelOp::EQ:
	out << "((" << print(rel.getLHS()) << ") == (" << print(rel.getRHS()) << "))";
	break;
      case BinaryRelOp::NE:
	out << "((" << print(rel.getLHS()) << ") != (" << print(rel.getRHS()) << "))";
	break;
      case BinaryRelOp::LT:
	out << "((" << print(rel.getLHS()) << ") < (" << print(rel.getRHS()) << "))";
	break;
      case BinaryRelOp::LE:
	out << "((" << print(rel.getLHS()) << ") <= (" << print(rel.getRHS()) << "))";
	break;
      case BinaryRelOp::GT:
	out << "((" << print(rel.getLHS()) << ") > (" << print(rel.getRHS()) << "))";
	break;
      case BinaryRelOp::GE:
	out << "((" << print(rel.getLHS()) << ") >= (" << print(rel.getRHS()) << "))";
	break;

	// strings
      case BinaryRelOp::MATCH: {
	out << "regex_wrapper(symTable.resolve((size_t)";
	out << print(rel.getLHS());
	out << "),symTable.resolve((size_t)";
	out << print(rel.getRHS());
	out << "))";
	break;
      }
      case BinaryRelOp::CONTAINS: {
	out << "(std::string(symTable.resolve((size_t)";
	out << print(rel.getRHS());
	out << ")).find(symTable.resolve((size_t)";
	out << print(rel.getLHS());
	out << "))!=std::string::npos)";
	break;
      }
      default:
	assert(0 && "unsupported operation");
	break;
      }
    }

    void visitEmpty(const RamEmpty& empty, irstream& out) {
      out << getRelationName(empty.getRelation()) << ".empty()";
    }

    void visitNotExists(const RamNotExists& ne, irstream& out) {

      // get some details
      const auto& rel = ne.getRelation();
      auto relName = getRelationName(rel);
      auto ctxName = "READ_OP_CONTEXT(" + getOpContextName(rel) + ")";
      auto arity = rel.getArity();

      // if it is total we use the contains function
      if (ne.isTotal()) {
	out << "!" << relName << ".contains(Tuple<RamDomain," << arity << ">({"
	    << join(ne.getValues(),",",rec)
	    << "})," << ctxName << ")";
	return;
      }

      // else we conduct a range query
      out << relName << ".equalRange";
      out << toIndex(ne.getKey());
      out << "(Tuple<RamDomain," << arity << ">({";
      out << join(ne.getValues(), ",", [&](irstream& out, RamValue* value) {
	  if (!value) out << "0";
	  else visit(*value, out);
	});
      out << "})," << ctxName << ").empty()";
    }

    // -- values --

    void visitNumber(const RamNumber& num, irstream& out) {
      out << num.getConstant();
    }

    void visitElementAccess(const RamElementAccess& access, irstream& out) {
      out << "env" << access.getLevel() << "[" << access.getElement() << "]";
    }

    void visitAutoIncrement(const RamAutoIncrement& inc, irstream& out) {
      out << "(ctr++)";
    }

    void visitBinaryOperator(const RamBinaryOperator& op, irstream& out) {
      switch (op.getOperator()) {

	// arithmetic
      case BinaryOp::ADD:
	out << "(" << print(op.getLHS()) << ") + (" << print(op.getRHS()) << ")";
	break;
      case BinaryOp::SUB:
	out << "(" << print(op.getLHS()) << ") - (" << print(op.getRHS()) << ")";
	break;
      case BinaryOp::MUL:
	out << "(" << print(op.getLHS()) << ") * (" << print(op.getRHS()) << ")";
	break;
      case BinaryOp::DIV:
	out << "(" << print(op.getLHS()) << ") / (" << print(op.getRHS()) << ")";
	break;
      case BinaryOp::EXP: {
	out << "(RamDomain)(std::pow((long)" << print(op.getLHS()) << "," << "(long)" << print(op.getRHS()) << "))";
	break;
      }
      case BinaryOp::MOD: {
	out << "(" << print(op.getLHS()) << ") % (" << print(op.getRHS()) << ")";
	break;
      }

	// strings
      case BinaryOp::CAT: {
	out << "(RamDomain)symTable.lookup(";
	out << "(std::string(symTable.resolve((size_t)";
	out << print(op.getLHS());
	out << ")) + std::string(symTable.resolve((size_t)";
	out << print(op.getRHS());
	out << "))).c_str())";
	break;
      }
      default:
	assert(0 && "unsupported operation");

      }
    }

    // -- records --

    void visitPack(const RamPack& pack, irstream& out) {
      out << "pack("
	  << "ram::Tuple<RamDomain," << pack.getValues().size() << ">({"
	  << join(pack.getValues(),",",rec)
	  << "})"
	  << ")";
    }

    void visitOrd(const RamOrd& ord, irstream& out) {
      out << print(ord.getSymbol());
    }

    void visitNegation(const RamNegation& neg, irstream& out) {
      out << "(-" << print(neg.getValue()) << ")"; 
    }

    // -- safety net --

    void visitNode(const RamNode& node, irstream&) {
      std::cout << "Unsupported node Type: " << typeid(node).name() << "\n";
      assert(false && "Unsupported Node Type!");
    }

  private:

    IRgencode print(const RamNode& node) {
      return IRgencode(*this, node);
    }

    IRgencode print(const RamNode* node) {
      return print(*node);
    }

    std::string getRelationName(const RamRelationIdentifier& rel) const {
      return "rel_" + rel.getName();
    }

    std::string getOpContextName(const RamRelationIdentifier& rel) const {
      return "rel_" + rel.getName() + "_op_ctxt";
    }
  };


  void genCode(irstream& out, const RamStatement& stmt, const RamExecutorConfig& config, const IndexMap& indices) {
    // use IRgencode
    IRgen(config, indices).visit(stmt,out);
  }

}


std::string RamIRCompiler::compileToIR(const SymbolTable& symTable, const RamStatement& stmt) const {

  // collect all used indices
  IndexMap indices;
  visitDepthFirst(stmt, [&](const RamNode& node) {
      if (const RamScan* scan = dynamic_cast<const RamScan*>(&node)) {
	indices[scan->getRelation()].addSearch(scan->getRangeQueryColumns());
      }
      if (const RamAggregate* agg = dynamic_cast<const RamAggregate*>(&node)) {
	indices[agg->getRelation()].addSearch(agg->getRangeQueryColumns());
      }
      if (const RamNotExists* ne = dynamic_cast<const RamNotExists*>(&node)) {
	indices[ne->getRelation()].addSearch(ne->getKey());
      }
    });

  // compute smallest number of indices (and report)
  if (report) *report << "------ Auto-Index-Generation Report -------\n";
  for(auto& cur : indices) {
    cur.second.solve();
    if (report) {
      *report << "Relation " << cur.first.getName() << "\n";
      *report << "\tNumber of Scan Patterns: " << cur.second.getSearches().size() << "\n";
      for(auto& cols : cur.second.getSearches()) {
	*report << "\t\t";
	for(uint32_t i=0;i<cur.first.getArity();i++) { 
	  if ((1UL<<i) & cols) {
	    *report << cur.first.getArg(i) << " "; 
	  }
	}
	*report << "\n";
      }
      *report << "\tNumber of Indexes: " << cur.second.getAllOrders().size() << "\n";
      for(auto& order : cur.second.getAllOrders()) {
	*report << "\t\t";
	for(auto& i : order) {
	  *report << cur.first.getArg(i) << " ";
	}
	*report << "\n";
      }
      *report << "------ End of Auto-Index-Generation Report -------\n";
    }
  }

  // ---------------------------------------------------------------
  //                      Code Generation
  // ---------------------------------------------------------------


  // open output file
  std::string fname;
  if (getBinaryFile() == "") {
    // generate temporary file
    char templ[40] = "./fileXXXXXX";
    close(mkstemp(templ));
    fname = templ;
  } else {
    fname = getBinaryFile();
  }

  // generate class name 
  char *bname = strdup(fname.c_str());
  std::string simplename = basename(bname);
  free(bname);
  for(size_t i=0;i<simplename.length();i++) {
    if((!isalpha(simplename[i]) && i == 0) || !isalnum(simplename[i]))  {
      simplename[i]='_';
    }
  }
  std::string classname = "Sf_" + simplename; 

  std::string binary = fname;
  std::string source = fname + ".cpp";

  // open output stream for header file
  std::ofstream os(source);

  // generate C++ program
  os << "#include \"souffle/CompiledSouffle.h\"\n";
  os << "\n";
  os << "namespace souffle {\n";
  os << "using namespace ram;\n";

  // print wrapper for regex
  os << "class " << classname << " : public Program {\n";
  os << "private:\n";
  os << "static bool regex_wrapper(const char *pattern, const char *text) {\n";
  os << "   bool result = false; \n";
  os << "   try { result = std::regex_match(text, std::regex(pattern)); } catch(...) { \n";
  os << "     std::cerr << \"warning: wrong pattern provided for match(\\\"\" << pattern << \"\\\",\\\"\" << text << \"\\\")\\n\";\n}\n";
  os << "   return result;\n";
  os << "}\n";
   
  if (getConfig().isLogging()) {
    os << "std::string profiling_fname;\n";
  }

  // declare symbol table
  os << "public:\n";
  os << "SymbolTable symTable;\n";
  os << "protected:\n";

  // print relation definitions
  std::string initCons; // initialization of constructor 
  std::string registerRel; // registration of relations 
  int relCtr=0;
  visitDepthFirst(stmt, [&](const RamCreate& create) {
      // get some table details
      const auto& rel = create.getRelation();
      auto type = getRelationType(rel.getArity(), indices[rel]);
      int arity = rel.getArity(); 
      const std::string &name = rel.getName(); 

      // defining table
      os << "// -- Table: " << name << "\n";
      os << type << " rel_" << name << ";\n";
      bool isTemp = (name.find("_temp1_")==0) || (name.find("_temp2_")==0);
      if ((rel.isInput() || rel.isComputed() || getConfig().isDebug()) && !isTemp) {
	os << "souffle::RelationWrapper<"; 
	os << relCtr++ << ",";
	os << type << ",";
	os << "Tuple<RamDomain," << arity << ">,";
	os << arity << ","; 
	os << (rel.isInput()?"true":"false") << ",";
	os << (rel.isComputed()?"true":"false");
	os << "> wrapper_" << name << ";\n"; 
          
	// construct types 
	std::string tupleType = "std::array<const char *," + std::to_string(arity) + ">{"; 
	tupleType += "\"" + rel.getArgTypeQualifier(0) + "\"";
	for(int i=1; i<arity; i++) {
	  tupleType += ",\"" + rel.getArgTypeQualifier(i) + "\"";
	}
	tupleType += "}";
	std::string tupleName = "std::array<const char *," + std::to_string(arity) + ">{";
	tupleName += "\"" + rel.getArg(0) + "\"";
	for (int i=1; i<arity; i++) {
	  tupleName += ",\"" + rel.getArg(i) + "\"";
	}
	tupleName += "}";
	if (initCons.size() > 0) { 
	  initCons += ",\n";
	}
	initCons += "wrapper_" + name + "(rel_" + name + ",symTable,\"" + name + "\"," + tupleType + "," + tupleName + ")";
	registerRel += "addRelation(\"" + name + "\",&wrapper_" + name + "," + std::to_string(rel.isInput()) + "," + std::to_string(rel.isOutput()) + ");\n";
      }
    });

  os << "public:\n";
    
  // -- constructor --

  os << classname;
  if (getConfig().isLogging()) {
    os << "(std::string pf=\"profile.log\") : profiling_fname(pf)";
    if (initCons.size() > 0) {
      os << ",\n";
    }
  } else {
    os << "() : \n";
  }
  os << initCons; 
  os << "{\n";
  os << registerRel; 

  if (symTable.size() > 0) {

    os << "// -- initialize symbol table --\n";
    os << "static const char *symbols[]={\n";
    for(size_t i=0;i<symTable.size();i++) {
      os << "\tR\"(" << symTable.resolve(i) << ")\",\n";
    }
    os << "};\n";
    os << "symTable.insert(symbols," << symTable.size() <<  ");\n";
    os << "\n";
  }

  os << "}\n";


  // -- run function --

  os << "void run() {\n";

  // initialize counter
  os << "// -- initialize counter --\n";
  os << "std::atomic<RamDomain> ctr(0);\n\n";

  // set default threads (in embedded mode)
  os << "#if defined(__EMBEDDED_SOUFFLE__) && defined(_OPENMP)\n";
  os << "omp_set_num_threads(" << getConfig().getNumThreads() << ");\n";
  os << "#endif\n\n";

  // add actual program body
  os << "// -- query evaluation --\n";
  if (getConfig().isLogging()) {
    os << "std::ofstream profile(profiling_fname);\n";
    os << "profile << \"@start-debug\\n\";\n";
    genCode(os, stmt, getConfig(), indices);
  } else {
    genCode(os, stmt, getConfig(), indices);
  }
  os << "}\n"; // end of run() method

  // issue printAll method
  os << "public:\n";
  os << "void printAll(std::string dirname=\"" << getConfig().getOutputDir() << "\") {\n";
  bool toConsole = (getConfig().getOutputDir() == "-");
  visitDepthFirst(stmt, [&](const RamStatement& node) {
      if (auto store = dynamic_cast<const RamStore*>(&node)) {
	auto name = store->getRelation().getName();
	auto relName = "rel_" + name;

	// pick target
	std::string fname = "dirname + \"/" + store->getFileName() + "\"";
	auto target = (toConsole) ? "nullptr" : fname;

	if (toConsole) {
	  os << "std::cout << \"---------------\\n" << name << "\\n===============\\n\";\n";
	}

	// create call
	os << relName << ".printCSV(" << target;
	os << ",symTable";

	// add format parameters
	const SymbolMask& mask = store->getSymbolMask();
	for(size_t i=0; i<store->getRelation().getArity(); i++) {
	  os << ((mask.isSymbol(i)) ? ",1" : ",0");
	}

	os << ");\n";

	if (toConsole) os << "std::cout << \"===============\\n\";\n";
      } else if (auto print = dynamic_cast<const RamPrintSize*>(&node)) {
	os << "{ auto lease = getOutputLock().acquire(); \n";
	os << "std::cout << R\"(" << print->getLabel() << ")\" <<  rel_" << print->getRelation().getName() << ".size() << \"\\n\";\n";
	os << "}";
      }
    });
  os << "}\n";  // end of printAll() method

  // issue loadAll method
  os << "public:\n";
  os << "void loadAll(std::string dirname=\"" << getConfig().getOutputDir() << "\") {\n";
  visitDepthFirst(stmt, [&](const RamLoad& load) {
      // get some table details
      os << "rel_";
      os << load.getRelation().getName();
      os << ".loadCSV(dirname + \"/";
      os << load.getFileName() << "\"";
      os << ",symTable";
      for(size_t i=0;i<load.getRelation().getArity();i++) {
	os << (load.getSymbolMask().isSymbol(i) ? ",1" : ",0");
      }
      os << ");\n";
    });
  os << "}\n";  // end of loadAll() method

  os << "public:\n";
  os << "const SymbolTable &getSymbolTable() const {\n";
  os << "return symTable;\n";
  os << "}\n"; // end of getSymbolTable() method

  os << "};\n"; // end of class declaration

  // factory base symbol (weak linkage: may be multiply defined)
  os << "ProgramFactory *ProgramFactory::base __attribute__ ((weak)) = nullptr;\n";

  // hidden hooks
  os << "Program *newInstance_" << simplename << "(){return new " << classname << ";}\n";
  os << "SymbolTable *getST_" << simplename << "(Program *p){return &reinterpret_cast<"
     << classname << "*>(p)->symTable;}\n";

  os << "#ifdef __EMBEDDED_SOUFFLE__\n";
  os << "class factory_" << classname << ": public souffle::ProgramFactory {\n";
  os << "Program *newInstance() {\n";
  os << "return new " << classname << "();\n";
  os << "};\n";
  os << "public:\n";
  os << "factory_" << classname << "() : ProgramFactory(\"" << fname << "\"){}\n";
  os << "};\n";
  os << "static factory_" << classname << " factory;\n";
  os << "}\n";
  os << "#else\n";
  os << "}\n";
  os << "int main(int argc, char** argv)\n{\n";

  // parse arguments
  os << "CmdOptions opt(" << getConfig().isLogging() << "," << getConfig().isDebug() << ");\n";
  os << "opt.analysis_src = R\"(" << getConfig().getSourceFileName() << ")\";\n";
  os << "opt.input_dir = R\"(" << getConfig().getFactFileDir() << ")\";\n";
  os << "opt.output_dir = R\"(" << getConfig().getOutputDir() << ")\";\n";
     
  if (getConfig().isLogging()) {
    os << "opt.profile_fname = R\"(" << getConfig().getProfileName() << ")\";\n";
  }
  os << "if (!opt.parse(argc,argv)) return 1;\n";

  os << "#if defined(_OPENMP) \n";
  os << "omp_set_nested(true);\n";
  os << "#endif\n";

  os << "souffle::";
  if (getConfig().isLogging()) {
    os << classname + " obj(opt.profile_fname);\n";
  } else {
    os << classname + " obj;\n";
  }

  os << "obj.loadAll(opt.input_dir);\n";
  os << "obj.run();\n";
  os << "if (!opt.output_dir.empty()) obj.printAll(opt.output_dir);\n";

  os << "return 0;\n";
  os << "}\n";
  os << "#endif\n";

  // close source file 
  os.close();


  // ---------------------------------------------------------------
  //                    Compilation & Execution
  // ---------------------------------------------------------------


  // execute shell script that compiles the generated C++ program
  std::string cmd = getConfig().getCompileScript(); 
  cmd += source;

  // set up number of threads
  auto num_threads = getConfig().getNumThreads();
  if (num_threads == 1) {
    cmd+=" seq";
  } else if (num_threads != 0) {
    cmd += " " + std::to_string(num_threads);
  }

  // separate souffle output form executable output
  if (getConfig().isLogging()) {
    std::cout.flush();
  }

  // run executable
  if(system(cmd.c_str()) != 0) {
    std::cerr << "failed to compile C++ source " << fname << "\n";
  }

  // done
  return binary;
}
