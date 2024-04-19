from typing import Set, Dict
import itertools
import print_x86defs
import sys
import traceback

from cs3020_support.python import *
import cs3020_support.x86 as x86
import constants
import cif
from interference_graph import InterferenceGraph

comparisons = ['eq', 'gt', 'gte', 'lt', 'lte']
gensym_num = 0
global_logging = False

tuple_var_types = {}
dataclass_var_types = {}
function_names = set()


def log(label, value):
    if global_logging:
        print()
        print(f'--------------------------------------------------')
        print(f'Logging: {label}')
        print(value)
        print(f'--------------------------------------------------')


def log_ast(label, value):
    log(label, print_ast(value))


def gensym(x):
    """
    Constructs a new variable name guaranteed to be unique.
    :param x: A "base" variable name (e.g. "x")
    :return: A unique variable name (e.g. "x_1")
    """

    global gensym_num
    gensym_num = gensym_num + 1
    return f'{x}_{gensym_num}'


##################################################
# typecheck
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr]) | Begin(Stmts, Expr)
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

TEnv = Dict[str, type]


@dataclass
class Callable:
    args: List[type]
    output_type: type

@dataclass
class DataclassType:
    name: str
    fields: Dict[str, any]
    field_types: Dict[str, type]


def typecheck(program: Program) -> Program:
    """
    Typechecks the input program; throws an error if the program is not well-typed.
    :param program: The Lfun program to typecheck
    :return: The program, if it is well-typed
    """

    prim_arg_types = {
        'add': [int, int],
        'sub': [int, int],
        'mult': [int, int],
        'not': [bool],
        'or': [bool, bool],
        'and': [bool, bool],
        'gt': [int, int],
        'gte': [int, int],
        'lt': [int, int],
        'lte': [int, int],
    }

    prim_output_types = {
        'add': int,
        'sub': int,
        'mult': int,
        'not': bool,
        'or': bool,
        'and': bool,
        'gt': bool,
        'gte': bool,
        'lt': bool,
        'lte': bool,
    }

    def tc_exp(e: Expr, env: TEnv) -> type:
        match e:
            case Var(x):
                return env[x]
            case Constant(i):
                if isinstance(i, bool):
                    return bool
                elif isinstance(i, int):
                    return int
                else:
                    raise Exception('tc_exp', e)
            case Call(e1, args):
                c_type = env[e1.name]
                assert isinstance(c_type, Callable)
                for i in range(len(args)):
                    assert tc_exp(args[i], env) == c_type.args[i]
                return c_type.output_type
            case FieldRef(e1, field_name):
                # Typecheck e1
                e1_type = tc_exp(e1, env)
                assert isinstance(e1_type, DataclassType)
                # Return the type that 'field_name' has in the DataclassType for e1
                return e1_type.field_types[field_name]
            case Begin(stmts, exp):
                tc_stmts(stmts, env)
                return tc_exp(exp, env)
            case Prim('eq', [e1, e2]):
                assert tc_exp(e1, env) == tc_exp(e2, env)
                return bool
            case Prim('tuple', args):
                arg_types = [tc_exp(arg, env) for arg in args]
                return tuple(arg_types)
            case Prim('subscript', [e1, Constant(i)]):
                t_e1 = tc_exp(e1, env)
                assert isinstance(t_e1, tuple)
                return t_e1[i]
            case Prim(op, args):
                arg_types = [tc_exp(a, env) for a in args]
                assert arg_types == prim_arg_types[op]
                return prim_output_types[op]
            case _:
                raise Exception('tc_exp', e)

    def tc_stmt(s: Stmt, env: TEnv):
        match s:
            case FunctionDef(name, args, body_stmts, return_type):
                # Add a binding to the original type environment of the form:
                # name -> Callable[[t1, ..., tk], return_type]
                arg_types = [a[1] for a in args]
                env[name] = Callable(arg_types, return_type)

                # Make a copy of the current env
                env_copy = {}
                for k, i in env.items():
                    env_copy[k] = i

                # Add a binding ai -> ti to the env copy for each argument ai and its type ti
                for i in range(len(args)):
                    env_copy[args[i][0]] = args[i][1]

                # Add a binding 'return_type' -> return_type to the env copy
                env_copy['return_type'] = return_type

                # Typecheck the body stmts using tc_stmt with the env copy
                tc_stmts(body_stmts, env_copy)

                # Add any tuples to tuple_var_types
                for v, t in env_copy.items():
                    if isinstance(t, tuple):
                        tuple_var_types[v] = t

                # Add name to the global set function_names to remember that it's a function name
                function_names.add(name)
            case ClassDef(name, superclass, body):
                # Create DataclassType and add it to global dictionary
                field_types = {}
                fields = {}
                for arg in body:
                    fields[arg[0]] = None
                    field_types[arg[0]] = arg[1]
                env[name] = Callable(args=field_types,
                                     output_type=DataclassType(name=name, fields=fields, field_types=field_types))
                return env[name]
            case While(condition, body):
                # Check condition is well-typed
                assert tc_exp(condition, env) == bool

                # Typcheck the body
                tc_stmts(body, env)
            case Return(e1):
                assert tc_exp(e1, env) == env['return_type']
            case If(condition, then_stmts, else_stmts):
                assert tc_exp(condition, env) == bool
                tc_stmts(then_stmts, env)
                tc_stmts(else_stmts, env)
            case Print(e):
                tc_exp(e, env)
            case Assign(x, e):
                t_e = tc_exp(e, env)
                if x in env:
                    assert t_e == env[x]
                else:
                    env[x] = t_e
            case _:
                raise Exception('tc_stmt', s)

    def tc_stmts(stmts: List[Stmt], env: TEnv):
        for s in stmts:
            tc_stmt(s, env)

    env = {}
    tc_stmts(program.stmts, env)
    for v, t in env.items():
        if isinstance(t, tuple):
            tuple_var_types[v] = t
        # Add dataclasses to global dictionary
        elif isinstance(t, DataclassType):
            dataclass_var_types[v] = t
    return program


##################################################
# remove-complex-opera*
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

def rco(prog: Program) -> Program:
    """
    Removes complex operands. After this pass, the arguments to operators (unary and binary
    operators, and function calls like "print") will be atomic.
    :param prog: An Lfun program
    :return: An Lfun program with atomic operator arguments.
    """

    def rco_stmt(stmt: Stmt, bindings: Dict[str, Expr]) -> Stmt:
        match stmt:
            case Assign(x, e1):
                new_e1 = rco_exp(e1, bindings)
                return Assign(x, new_e1)
            case Print(e1):
                new_e1 = rco_exp(e1, bindings)
                return Print(new_e1)
            case While(condition, body_stmts):
                condition_bindings = {}
                condition_exp = rco_exp(condition, condition_bindings)
                condition_stmts = [Assign(x, e) for x, e in condition_bindings.items()]
                new_condition = Begin(condition_stmts, condition_exp)

                new_body_stmts = rco_stmts(body_stmts)
                return While(new_condition, new_body_stmts)

            case If(condition, then_stmts, else_stmts):
                new_condition = rco_exp(condition, bindings)
                new_then_stmts = rco_stmts(then_stmts)
                new_else_stmts = rco_stmts(else_stmts)

                return If(new_condition,
                          new_then_stmts,
                          new_else_stmts)
            case FunctionDef(name, args, body_stmts, return_type):
                new_stmts = rco_stmts(body_stmts)
                return FunctionDef(name, args, new_stmts, return_type)
            case Return(e1):
                return Return(rco_exp(e1, bindings))
            case _:
                raise Exception('rco_stmt', stmt)

    def rco_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []

        for stmt in stmts:
            bindings = {}
            # (1) compile the statement
            new_stmt = rco_stmt(stmt, bindings)
            # (2) add the new bindings created by rco_exp
            new_stmts.extend([Assign(x, e) for x, e in bindings.items()])
            # (3) add the compiled statement itself
            new_stmts.append(new_stmt)

        return new_stmts

    def rco_exp(e: Expr, bindings: Dict[str, Expr]) -> Expr:
        match e:
            case Var(x):
                if x in function_names:
                    # Generate a tmp for it
                    new_x = gensym('tmp')
                    bindings[new_x] = e
                    return Var(new_x)
                else:
                    return Var(x)
            case Constant(i):
                return Constant(i)
            case Call(e1, args):
                new_args = [rco_exp(e, bindings) for e in args]
                new_name = rco_exp(e1, bindings)
                new_e = Call(new_name, new_args)
                new_v = gensym('tmp')
                bindings[new_v] = new_e
                return Var(new_v)
            case Prim(op, args):
                new_args = [rco_exp(e, bindings) for e in args]
                new_e = Prim(op, new_args)
                new_v = gensym('tmp')
                bindings[new_v] = new_e
                return Var(new_v)
            case _:
                raise Exception('rco_exp', e)

    return Program(rco_stmts(prog.stmts))

##################################################
# compile dataclasses
##################################################
def compile_dataclasses(prog: Program) -> Program:
    return prog


##################################################
# explicate-control
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "subscript" | "allocate" | "collect" | "tuple_set"
# Atm    ::= Var(x) | Constant(n)
# Expr   ::= Atm | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

def explicate_control(prog: Program) -> cif.CProgram:
    """
    Transforms an Lfun Expression into a Cif program.
    :param prog: An Lfun Expression
    :return: A Cif Program
    """
    main_stmts = []
    defs: List[cif.CFunctionDef] = []
    stmts = prog.stmts  # Statements of the Python program
    for s in stmts:
        match s:
            case FunctionDef(name, args, body, return_type):
                # Call _explicate_control on the body, treating it like a mini-program
                # Call `_explicate_control` on the `body_stmts` of the function, pretending it is a whole Python program
                blocks = _explicate_control(name, Program(body))
                fun_def = cif.CFunctionDef(name, [a[0] for a in args], blocks)
                defs.append(fun_def)
            case _:
                main_stmts.append(s)

    # Call _explicate_control on main_stmts, treating it like a mini-program for the "main" function
    main_blocks = _explicate_control("main", Program(main_stmts))
    main_def = cif.CFunctionDef("main", [], main_blocks)
    return cif.CProgram(defs + [main_def])


def _explicate_control(current_function: str, prog: Program) -> Dict[str, List[cif.Stmt]]:
    """
    Transforms an Lif Expression into a Cif program.
    :param prog: An Lif Expression
    :return: A Cif Program
    """

    # the basic blocks of the program
    basic_blocks: Dict[str, List[cif.Stmt]] = {}

    # create a new basic block to hold some statements
    # generates a brand-new name for the block and returns it
    def create_block(stmts: List[cif.Stmt]) -> str:
        label = gensym('label')
        basic_blocks[label] = stmts
        return label

    def explicate_atm(e: Expr) -> cif.Atm:
        match e:
            case Var(x):
                return cif.Var(x)
            case Constant(c):
                return cif.Constant(c)
            case _:
                raise RuntimeError(e)

    def explicate_exp(e: Expr) -> cif.Expr:
        match e:
            case Prim(op, args):
                new_args = [explicate_atm(a) for a in args]
                return cif.Prim(op, new_args)
            case Call(e1, args):
                new_args = [explicate_atm(a) for a in args]
                new_exp = explicate_atm(e1)
                return cif.Call(new_exp, new_args)
            case _:
                return explicate_atm(e)

    def explicate_stmt(stmt: Stmt, cont: List[cif.Stmt]) -> List[cif.Stmt]:
        match stmt:
            case Assign(x, exp):
                new_exp = explicate_exp(exp)
                new_stmt: List[cif.Stmt] = [cif.Assign(x, new_exp)]
                return new_stmt + cont
            case Print(exp):
                new_exp = explicate_atm(exp)
                new_stmt: List[cif.Stmt] = [cif.Print(new_exp)]
                return new_stmt + cont
            case While(Begin(condition_stmts, condition_exp), body_stmts):
                cont_label = create_block(cont)
                test_label = gensym('loop_label')
                body_label = create_block(explicate_stmts(body_stmts, [cif.Goto(test_label)]))
                test_stmts = [cif.If(explicate_exp(condition_exp),
                                     cif.Goto(body_label),
                                     cif.Goto(cont_label))]
                basic_blocks[test_label] = explicate_stmts(condition_stmts, test_stmts)
                return [cif.Goto(test_label)]
            case If(condition, then_stmts, else_stmts):
                cont_label = create_block(cont)
                e2_label = create_block(explicate_stmts(then_stmts, [cif.Goto(cont_label)]))
                e3_label = create_block(explicate_stmts(else_stmts, [cif.Goto(cont_label)]))
                return [cif.If(explicate_exp(condition),
                               cif.Goto(e2_label),
                               cif.Goto(e3_label))]
            case Return(e1):
                new_exp = explicate_atm(e1)
                return [cif.Return(new_exp)]
            case _:
                raise RuntimeError(stmt)

    def explicate_stmts(stmts: List[Stmt], cont: List[cif.Stmt]) -> List[cif.Stmt]:
        for s in reversed(stmts):
            cont = explicate_stmt(s, cont)
        return cont

    new_body = [cif.Return(cif.Constant(0))]
    new_body = explicate_stmts(prog.stmts, new_body)

    basic_blocks[current_function + 'start'] = new_body
    return basic_blocks


##################################################
# select-instructions
##################################################
# op           ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#                | "subscript" | "allocate" | "collect" | "tuple_set"
# Atm          ::= Var(x) | Constant(n)
# Expr         ::= Atm | Prim(op, List[Expr])
# Stmt         ::= Assign(x, Expr) | Print(Expr)
#                | If(Expr, Goto(label), Goto(label)) | Goto(label) | Return(Expr)
# Stmts        ::= List[Stmt]
# CFunctionDef ::= CFunctionDef(name, List[str], Dict[label, Stmts])
# Cif         ::= CProgram(List[CFunctionDef])

@dataclass(frozen=True, eq=True)
class X86FunctionDef(AST):
    label: str
    blocks: Dict[str, List[x86.Instr]]
    stack_space: Tuple[int, int]


@dataclass(frozen=True, eq=True)
class X86ProgramDefs(AST):
    defs: List[X86FunctionDef]


def select_instructions(prog: cif.CProgram) -> X86ProgramDefs:
    """
    Transforms a Lfun program into a pseudo-x86 assembly program.
    :param prog: a Lfun program
    :return: a pseudo-x86 program
    """
    program_defs = X86ProgramDefs(defs=[])
    for fun in prog.defs:
        match fun:
            case cif.CFunctionDef(name, args, blocks):
                func_blocks = _select_instructions(name, blocks).blocks

                # Moving arguments into parameter variables
                params = []
                for i in range(len(args)):
                    instr = x86.NamedInstr('movq', [x86.Reg(constants.argument_registers[i]), x86.Var(args[i])])
                    func_blocks[name + 'start'].insert(i, instr)

                # Make a function def for program defs
                program_defs.defs.append(X86FunctionDef(label=name, blocks=func_blocks, stack_space=None))
            case _:
                raise Exception('si func', fun)
    return program_defs


def _select_instructions(current_function: str, blocks: Dict[str, cif.Stmt]) -> x86.X86Program:
    """
    Transforms a Lif program into a pseudo-x86 assembly program.
    :param prog: a Lif program
    :return: a pseudo-x86 program
    """

    def mk_tag(types: Tuple[type]) -> int:
        """
        Builds a vector tag. See section 5.2.2 in the textbook.
        :param types: A list of the types of the vector's elements.
        :return: A vector tag, as an integer.
        """
        pointer_mask = 0
        # for each type in the vector, encode it in the pointer mask
        for t in reversed(types):
            # shift the mask by 1 bit to make room for this type
            pointer_mask = pointer_mask << 1

            if isinstance(t, tuple):
                # if it's a vector type, the mask is 1
                pointer_mask = pointer_mask + 1
            else:
                # otherwise, the mask is 0 (do nothing)
                pass

        # shift the pointer mask by 6 bits to make room for the length field
        mask_and_len = pointer_mask << 6
        mask_and_len = mask_and_len + len(types)  # add the length

        # shift the mask and length by 1 bit to make room for the forwarding bit
        tag = mask_and_len << 1
        tag = tag + 1

        return tag

    def si_atm(a: cif.Expr) -> x86.Arg:
        match a:
            case cif.Constant(i):
                return x86.Immediate(int(i))
            case cif.Var(x):
                if x in function_names:
                    return x86.GlobalVal(x)
                else:
                    return x86.Var(x)
            case _:
                raise Exception('si_atm', a)

    def si_stmts(stmts: List[cif.Stmt]) -> List[x86.Instr]:
        instrs = []

        for stmt in stmts:
            instrs.extend(si_stmt(stmt))

        return instrs

    op_cc = {'eq': 'e', 'gt': 'g', 'gte': 'ge', 'lt': 'l', 'lte': 'le'}

    binop_instrs = {'add': 'addq', 'sub': 'subq', 'mult': 'imulq', 'and': 'andq', 'or': 'orq'}

    def si_stmt(stmt: cif.Stmt) -> List[x86.Instr]:
        match stmt:
            case cif.Assign(x, cif.Call(f, args)):
                instrs = []
                # Save caller-saved registers
                used_regs = []
                for reg in constants.caller_saved_registers:
                    instrs.append(x86.NamedInstr('pushq', [x86.Reg(reg)]))
                # Move the arguments into parameter-passing registers
                for i in range(len(args)):
                    instrs.append(x86.NamedInstr('movq', [si_atm(args[i]), x86.Reg(constants.argument_registers[i])]))
                # Add an indirect call node
                instrs.append(x86.IndirectCallq(si_atm(f), 0))
                # Loop backwards in the used registers to pop them off stack
                for i in range(len(constants.caller_saved_registers) - 1, -1, -1):
                    instrs.append(x86.NamedInstr('popq', [x86.Reg(constants.caller_saved_registers[i])]))
                # Move rax to x
                instrs.append(x86.NamedInstr('movq', [x86.Reg('rax'), x86.Var(x)]))
                return instrs
            case cif.Assign(x, cif.Prim('tuple', args)):
                tag = mk_tag(tuple_var_types[x])
                instrs = [x86.NamedInstr('movq', [x86.Immediate(8 * (1 + len(args))), x86.Reg('rdi')]),
                          x86.Callq('allocate'),
                          x86.NamedInstr('movq', [x86.Reg('rax'), x86.Reg('r11')]),
                          x86.NamedInstr('movq', [x86.Immediate(tag), x86.Deref('r11', 0)])]
                for i, a in enumerate(args):
                    instrs.append(x86.NamedInstr('movq', [si_atm(a), x86.Deref('r11', 8 * (i + 1))]))
                instrs.append(x86.NamedInstr('movq', [x86.Reg('r11'), x86.Var(x)]))
                return instrs
            case cif.Assign(x, cif.Prim('subscript', [atm1, cif.Constant(idx)])):
                offset_bytes = 8 * (idx + 1)
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('r11')]),
                        x86.NamedInstr('movq', [x86.Deref('r11', offset_bytes), x86.Var(x)])]
            case cif.Assign(x, cif.Prim(op, [atm1, atm2])):
                if op in binop_instrs:
                    return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                            x86.NamedInstr(binop_instrs[op], [si_atm(atm2), x86.Reg('rax')]),
                            x86.NamedInstr('movq', [x86.Reg('rax'), x86.Var(x)])]
                elif op in op_cc:
                    return [x86.NamedInstr('cmpq', [si_atm(atm2), si_atm(atm1)]),
                            x86.Set(op_cc[op], x86.ByteReg('al')),
                            x86.NamedInstr('movzbq', [x86.ByteReg('al'), x86.Var(x)])]
                else:
                    raise Exception('si_stmt failed op', op)
            case cif.Assign(x, cif.Prim('not', [atm1])):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Var(x)]),
                        x86.NamedInstr('xorq', [x86.Immediate(1), x86.Var(x)])]
            case cif.Assign(x, atm1):
                # Check if the atm1 is a function name
                atm = si_atm(atm1)
                if isinstance(atm, x86.GlobalVal):
                    return [x86.NamedInstr('leaq', [atm, x86.Var(x)])]
                else:
                    return [x86.NamedInstr('movq', [atm, x86.Var(x)])]
            case cif.Print(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rdi')]),
                        x86.Callq('print_int')]
            case cif.Return(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                        x86.Jmp(current_function + 'conclusion')]
            case cif.Goto(label):
                return [x86.Jmp(label)]
            case cif.If(a, cif.Goto(then_label), cif.Goto(else_label)):
                return [x86.NamedInstr('cmpq', [si_atm(a), x86.Immediate(1)]),
                        x86.JmpIf('e', then_label),
                        x86.Jmp(else_label)]
            case cif.Return(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                        x86.Jmp(current_function + 'conclusion')]
            case _:
                raise Exception('si_stmt', stmt)

    basic_blocks = {label: si_stmts(block) for (label, block) in blocks.items()}
    return x86.X86Program(basic_blocks)


##################################################
# allocate-registers
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

Color = int
Coloring = Dict[x86.Var, Color]
Saturation = Set[Color]


def allocate_registers(program: X86ProgramDefs) -> X86ProgramDefs:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param program: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """
    func_defs: List[X86FunctionDef] = []
    for d in program.defs:
        match d:
            case X86FunctionDef(label, blocks, stack_space):
                p = x86.X86Program(blocks, stack_space)
                result = _allocate_registers(label, p)
                func_defs.append(X86FunctionDef(label=label, blocks=result.blocks, stack_space=result.stack_space))
            case _:
                raise Exception('ar func', d)
    return X86ProgramDefs(defs=func_defs)


def _allocate_registers(current_function: str, program: x86.X86Program) -> x86.X86Program:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param program: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """

    blocks = program.blocks
    live_before_sets = {current_function + 'conclusion': set()}
    for label in blocks:
        live_before_sets[label] = set()

    live_after_sets = {}
    homes: Dict[x86.Var, x86.Arg] = {}
    tuple_homes: Dict[x86.Var, x86.Arg] = {}

    tuple_vars = set(tuple_var_types.keys())

    # --------------------------------------------------
    # utilities
    # --------------------------------------------------
    def vars_arg(a: x86.Arg) -> Set[x86.Var]:
        match a:
            case x86.Immediate(i):
                return set()
            case x86.GlobalVal(g):
                return set()
            case x86.Reg(r):
                return set()
            case x86.ByteReg(r):
                return set()
            case x86.Var(x):
                if x in tuple_var_types:
                    return set()
                else:
                    return {x86.Var(x)}
            case x86.Deref(r, offset):
                return set()
            case _:
                raise Exception('ul_arg', a)

    def reads_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, [e1, e2]) if i in ['movq', 'movzbq', 'leaq']:
                return vars_arg(e1)
            case x86.NamedInstr(i, [e1, e2]) if i in ['addq', 'subq', 'imulq', 'cmpq', 'andq', 'orq', 'xorq']:
                return vars_arg(e1).union(vars_arg(e2))
            case x86.Jmp(label) | x86.JmpIf(_, label):
                # the variables that might be read after this instruction
                # are the live-before variables of the destination block
                return live_before_sets[label]
            case x86.NamedInstr(i, args) if i in ['pushq', 'popq']:
                return set()
            case _:
                if isinstance(i, (x86.IndirectCallq, x86.Callq, x86.Set)):
                    return set()
                else:
                    raise Exception(i)

    def writes_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, args) if i in ['pushq', 'popq']:
                return set()
            case x86.NamedInstr(i, [e1, e2]) \
                if i in ['movq', 'movzbq', 'addq', 'subq', 'imulq', 'cmpq', 'andq', 'orq', 'xorq', 'leaq']:
                return vars_arg(e2)
            case _:
                if isinstance(i, (x86.Jmp, x86.JmpIf, x86.Callq, x86.IndirectCallq, x86.Set)):
                    return set()
                else:
                    raise Exception(i)

    # --------------------------------------------------
    # liveness analysis
    # --------------------------------------------------
    def ul_instr(i: x86.Instr, live_after: Set[x86.Var]) -> Set[x86.Var]:
        return live_after.difference(writes_of(i)).union(reads_of(i))

    def ul_block(label: str):
        instrs = blocks[label]
        current_live_after: Set[x86.Var] = set()

        block_live_after_sets = []
        for i in reversed(instrs):
            block_live_after_sets.append(current_live_after)
            current_live_after = ul_instr(i, current_live_after)

        live_before_sets[label] = current_live_after
        live_after_sets[label] = list(reversed(block_live_after_sets))

    def ul_fixpoint(labels: List[str]):
        fixpoint_reached = False

        while not fixpoint_reached:
            old_live_befores = live_before_sets.copy()

            for label in labels:
                ul_block(label)

            if old_live_befores == live_before_sets:
                fixpoint_reached = True

    # --------------------------------------------------
    # interference graph
    # --------------------------------------------------
    def bi_instr(e: x86.Instr, live_after: Set[x86.Var], graph: InterferenceGraph):
        for v1 in writes_of(e):
            for v2 in live_after:
                graph.add_edge(v1, v2)

    def bi_block(instrs: List[x86.Instr], live_afters: List[Set[x86.Var]], graph: InterferenceGraph):
        for instr, live_after in zip(instrs, live_afters):
            bi_instr(instr, live_after, graph)

    # --------------------------------------------------
    # graph coloring
    # --------------------------------------------------
    def color_graph(local_vars: Set[x86.Var], interference_graph: InterferenceGraph) -> Coloring:
        coloring: Coloring = {}

        to_color = local_vars.copy()

        # Saturation sets start out empty
        saturation_sets = {x: set() for x in local_vars}

        # Loop until we are finished coloring
        while to_color:
            # Find the variable x with the largest saturation set
            x = max(to_color, key=lambda x: len(saturation_sets[x]))

            # Remove x from the variables to color
            to_color.remove(x)

            # Find the smallest color not in x's saturation set
            x_color = next(i for i in itertools.count() if i not in saturation_sets[x])

            # Assign x's color
            coloring[x] = x_color

            # Add x's color to the saturation sets of its neighbors
            for y in interference_graph.neighbors(x):
                saturation_sets[y].add(x_color)

        return coloring

    # --------------------------------------------------
    # assigning homes
    # --------------------------------------------------
    def align(num_bytes: int) -> int:
        if num_bytes % 16 == 0:
            return num_bytes
        else:
            return num_bytes + (16 - (num_bytes % 16))

    def ah_arg(a: x86.Arg) -> x86.Arg:
        match a:
            case x86.Immediate(i):
                return a
            case x86.GlobalVal(g):
                return a
            case x86.Deref(r, offset):
                return a
            case x86.Reg(r):
                return a
            case x86.ByteReg(r):
                return a
            case x86.Var(x):
                if x in tuple_vars:
                    if x in tuple_homes:
                        return tuple_homes[x]
                    else:
                        current_stack_size = len(tuple_homes) * 8
                        offset = -(current_stack_size + 8)
                        tuple_homes[x] = x86.Deref('r15', offset)
                        return x86.Deref('r15', offset)
                else:
                    if a in homes:
                        return homes[a]
                    else:
                        return x86.Reg('r8')
            case _:
                raise Exception('ah_arg', a)

    def ah_instr(e: x86.Instr) -> x86.Instr:
        match e:
            case x86.NamedInstr(i, args):
                return x86.NamedInstr(i, [ah_arg(a) for a in args])
            case x86.IndirectCallq(a1, n):
                return x86.IndirectCallq(ah_arg(a1), n)
            case x86.Set(cc, a1):
                return x86.Set(cc, ah_arg(a1))
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf)):
                    return e
                else:
                    raise Exception('ah_instr', e)

    def ah_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        return [ah_instr(i) for i in instrs]

    # --------------------------------------------------
    # main body of the pass
    # --------------------------------------------------

    # Step 1: Perform liveness analysis
    all_labels = list(blocks.keys())
    ul_fixpoint(all_labels)
    log_ast('live-after sets', live_after_sets)

    # Step 2: Build the interference graph
    interference_graph = InterferenceGraph()

    for label in blocks.keys():
        bi_block(blocks[label], live_after_sets[label], interference_graph)

    log_ast('interference graph', interference_graph)

    # Step 3: Color the graph
    all_vars = interference_graph.get_nodes()
    coloring = color_graph(all_vars, interference_graph)
    colors_used = set(coloring.values())
    log('coloring', coloring)

    # Defines the set of registers to use
    available_registers = constants.caller_saved_registers + constants.callee_saved_registers

    # Step 4: map variables to homes
    color_map = {}
    stack_locations_used = 0

    # Step 4.1: Map colors to locations (the "color map")
    for color in sorted(colors_used):
        if available_registers != []:
            r = available_registers.pop()
            color_map[color] = x86.Reg(r)
        else:
            offset = stack_locations_used + 1
            color_map[color] = x86.Deref('rbp', -(offset * 8))
            stack_locations_used += 1

    # Step 4.2: Compose the "coloring" with the "color map" to get "homes"
    for v in all_vars:
        homes[v] = color_map[coloring[v]]
    log('homes', homes)

    # Step 5: replace variables with their homes
    blocks = program.blocks
    new_blocks = {label: ah_block(block) for label, block in blocks.items()}
    regular_stack_space = align(8 * stack_locations_used)
    root_stack_slots = len(tuple_homes)
    return x86.X86Program(new_blocks, stack_space=(regular_stack_space, root_stack_slots))


##################################################
# patch-instructions
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

def patch_instructions(program: X86ProgramDefs) -> X86ProgramDefs:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param program: An x86 program.
    :return: A patched x86 program.
    """

    func_defs: List[X86FunctionDef] = []
    for d in program.defs:
        match d:
            case X86FunctionDef(label, blocks, stack_space):
                p = x86.X86Program(blocks, stack_space)
                result = _patch_instructions(p)
                func_defs.append(X86FunctionDef(label=label, blocks=result.blocks, stack_space=result.stack_space))
            case _:
                raise Exception('pi func', d)
    return X86ProgramDefs(defs=func_defs)


def _patch_instructions(program: x86.X86Program) -> x86.X86Program:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param program: An x86 program.
    :return: A patched x86 program.
    """

    def pi_instr(e: x86.Instr) -> List[x86.Instr]:
        match e:
            case x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                return [x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                        x86.NamedInstr('movq', [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case x86.NamedInstr('cmpq', [a1, x86.Immediate(i)]):
                return [x86.NamedInstr('movq', [x86.Immediate(i), x86.Reg('rax')]),
                        x86.NamedInstr('cmpq', [a1, x86.Reg('rax')])]
            case x86.NamedInstr(i, [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                return [x86.NamedInstr('movq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                        x86.NamedInstr(i, [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf, x86.NamedInstr, x86.Set, x86.IndirectCallq)):
                    return [e]
                else:
                    raise Exception('pi_instr', e)

    def pi_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = [pi_instr(i) for i in instrs]
        flattened = [val for sublist in new_instrs for val in sublist]
        return flattened

    blocks = program.blocks
    new_blocks = {label: pi_block(block) for label, block in blocks.items()}
    return x86.X86Program(new_blocks, stack_space=program.stack_space)


##################################################
# prelude-and-conclusion
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Deref(r, offset) | GlobalVal(x)
# op     ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#          | 'leaq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#          | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)

#          | IndirectCallq(Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

def prelude_and_conclusion(program: X86ProgramDefs) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param program: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """
    all_blocks = {}
    for d in program.defs:
        match d:
            case X86FunctionDef(label, blocks, stack_space):
                p = x86.X86Program(blocks, stack_space)
                result = _prelude_and_conclusion(label, p)
                for k, v in result.blocks.items():
                    all_blocks[k] = v
    return x86.X86Program(blocks=all_blocks)


def _prelude_and_conclusion(current_function: str, program: x86.X86Program) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param program: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """
    stack_bytes, root_stack_locations = program.stack_space

    # Construct prelude
    prelude = [x86.NamedInstr('pushq', [x86.Reg('rbp')]),
               x86.NamedInstr('movq', [x86.Reg('rsp'), x86.Reg('rbp')])]
    for reg in constants.callee_saved_registers:
        prelude.append(x86.NamedInstr('pushq', [x86.Reg(reg)]))
    prelude += [x86.NamedInstr('subq', [x86.Immediate(stack_bytes),
                                        x86.Reg('rsp')])]

    if current_function == 'main':
        prelude += [x86.NamedInstr('movq', [x86.Immediate(constants.root_stack_size),
                                            x86.Reg('rdi')]),
                    x86.NamedInstr('movq', [x86.Immediate(constants.heap_size),
                                            x86.Reg('rsi')]),
                    x86.Callq('initialize'),
                    x86.NamedInstr('movq', [x86.GlobalVal('rootstack_begin'), x86.Reg('r15')])]
    for offset in range(root_stack_locations):
        prelude += [x86.NamedInstr('movq', [x86.Immediate(0), x86.Deref('r15', 0)]),
                    x86.NamedInstr('addq', [x86.Immediate(8), x86.Reg('r15')])]
    prelude += [x86.Jmp(current_function + 'start')]

    # Construct conclusion
    conclusion = [x86.NamedInstr('addq', [x86.Immediate(stack_bytes),
                                          x86.Reg('rsp')]),
                  x86.NamedInstr('subq', [x86.Immediate(8 * root_stack_locations), x86.Reg('r15')])]
    for reg in reversed(constants.callee_saved_registers):
        conclusion.append(x86.NamedInstr('popq', [x86.Reg(reg)]))
    conclusion += [x86.NamedInstr('popq', [x86.Reg('rbp')]),
                   x86.Retq()]

    new_blocks = program.blocks.copy()
    new_blocks[current_function] = prelude
    new_blocks[current_function + 'conclusion'] = conclusion
    return x86.X86Program(new_blocks, stack_space=program.stack_space)


##################################################
# add-allocate
##################################################

def add_allocate(program: str) -> str:
    """
    Adds the 'allocate' function to the end of the program.
    :param program: An x86 program, as a string.
    :return: An x86 program, as a string, with the 'allocate' function.
    """

    allocate_label = ("allocate:\n"
                      "  movq free_ptr(%rip), %rax\n"
                      "  addq %rdi, %rax\n"
                      "  movq %rdi, %rsi\n"
                      "  cmpq fromspace_end(%rip), %rax\n"
                      "  jl allocate_alloc\n"
                      "  movq %r15, %rdi\n"
                      "  callq collect\n"
                      "  jmp allocate_alloc")

    allocate_alloc_label = ("allocate_alloc:\n"
                            "  movq free_ptr(%rip), %rax\n"
                            "  addq %rsi, free_ptr(%rip)\n"
                            "  retq\n")

    return program + allocate_label + "\n" + allocate_alloc_label


##################################################
# Compiler definition
##################################################

compiler_passes = {
    'typecheck': typecheck,
    'remove complex opera*': rco,
    'typecheck2': typecheck,
    'compile dataclasses': compile_dataclasses,
    'explicate control': explicate_control,
    'select instructions': select_instructions,
    'allocate registers': allocate_registers,
    'patch instructions': patch_instructions,
    'prelude & conclusion': prelude_and_conclusion,
    'print x86': x86.print_x86,
    'add allocate': add_allocate
}


def run_compiler(s, logging=False):
    global tuple_var_types, function_names
    tuple_var_types = {}
    function_names = set()

    def print_prog(current_program):
        print('Concrete syntax:')
        if isinstance(current_program, x86.X86Program):
            print(x86.print_x86(current_program))
        # BEGIN NEW CODE TO FIX THE BUG
        elif isinstance(current_program, X86ProgramDefs):
            print(print_x86defs.print_x86_defs(current_program))
        # END NEW CODE
        elif isinstance(current_program, Program):
            print(print_program(current_program))
        elif isinstance(current_program, cif.CProgram):
            print(cif.print_program(current_program))

        print()
        print('Abstract syntax:')
        print(print_ast(current_program))

    current_program = parse(s)

    if logging == True:
        print()
        print('==================================================')
        print(' Input program')
        print('==================================================')
        print()
        print_prog(current_program)

    for pass_name, pass_fn in compiler_passes.items():
        current_program = pass_fn(current_program)

        if logging == True:
            print()
            print('==================================================')
            print(f' Output of pass: {pass_name}')
            print('==================================================')
            print()
            print_prog(current_program)

    return current_program


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python compiler.py <source filename>')
    else:
        file_name = sys.argv[1]
        with open(file_name) as f:
            print(f'Compiling program {file_name}...')

            try:
                program = f.read()
                x86_program = run_compiler(program, logging=True)

                with open(file_name + '.s', 'w') as output_file:
                    output_file.write(x86_program)

            except:
                print('Error during compilation! **************************************************')
                traceback.print_exception(*sys.exc_info())
