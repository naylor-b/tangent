# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Control flow graph analysis.

Given a Python AST we construct a doubly linked control flow graph whose nodes
contain the AST of the statements. We can then perform forward analysis on this
CFG.

"""
from __future__ import absolute_import
import functools
import operator

import gast

from tangent import annotations as anno
from tangent import ast as ast_
from tangent import grammar
from tangent import utils


class Node(object):
  """A node in the CFG."""
  __slots__ = ['next', 'value', 'prev']

  def __init__(self, value):
    self.next = set()
    self.prev = set()
    self.value = value


class CFG(gast.NodeVisitor):
  """Construct a control flow graph.

  Each statement is represented as a node. For control flow statements such
  as conditionals and loops the conditional itself is a node which either
  branches or cycles, respectively.

  Attributes:
    entry: The entry node, which contains the `gast.arguments` node of the
        function definition.
    exit: The exit node. This node is special because it has no value (i.e. no
        corresponding AST node). This is because Python functions can have
        multiple return statements.
  """

  def __init__(self):
    # The current leaves of the CFG
    self.head = []
    # A stack of continue statements
    self.continue_ = []
    # A stack of break nodes
    self.break_ = []

  @staticmethod
  def backlink(node):
    """Given a CFG with outgoing links, create incoming links."""
    seen = set()
    to_see = [node]
    while to_see:
      node = to_see.pop()
      seen.add(node)
      for succ in node.next:
        succ.prev.add(node)
        if succ not in seen:
          to_see.append(succ)

  def set_head(self, node):
    """Link this node to the current leaves."""
    for head in self.head:
      head.next.add(node)
    self.head[:] = [node]

  @classmethod
  def build_cfg(cls, node):
    """Build a CFG for a function.

    Args:
      node: A function definition the body of which to analyze.

    Returns:
      A CFG object.

    Raises:
      TypeError: If the input is not a function definition.
    """
    if not isinstance(node, gast.FunctionDef):
      raise TypeError('input must be a function definition')
    cfg = cls()
    cfg.entry = Node(node.args)
    cfg.head = [cfg.entry]
    cfg.visit_statements(node.body)
    cfg.exit = Node(None)
    cfg.set_head(cfg.exit)
    cfg.backlink(cfg.entry)
    return cfg

  def visit_statements(self, nodes):
    for node in nodes:
      if isinstance(node, grammar.CONTROL_FLOW):
        self.visit(node)
      else:
        expr = Node(node)
        self.set_head(expr)

  def generic_visit(self, node):
    raise ValueError('unknown control flow')

  def visit_If(self, node):
    # The current head will hold the conditional
    test = Node(node.test)
    self.set_head(test)
    # Handle the body
    self.visit_statements(node.body)
    body_exit = self.head[:]
    self.head[:] = []
    self.head.append(test)
    # Handle the orelse
    self.visit_statements(node.orelse)
    self.head.extend(body_exit)

  def visit_While(self, node):
    test = Node(node.test)
    self.set_head(test)
    # Start a new level of nesting
    self.break_.append([])
    self.continue_.append([])
    # Handle the body
    self.visit_statements(node.body)
    self.head.extend(self.continue_.pop())
    self.set_head(test)
    # Handle the orelse
    self.visit_statements(node.orelse)
    # The break statements and the test go to the next node
    self.head.extend(self.break_.pop())

  def visit_For(self, node):
    iter_ = Node(node)
    self.set_head(iter_)
    self.break_.append([])
    self.continue_.append([])
    self.visit_statements(node.body)
    self.head.extend(self.continue_.pop())
    self.set_head(iter_)
    self.head.extend(self.break_.pop())

  def visit_Break(self, node):
    self.break_[-1].extend(self.head)
    self.head[:] = []

  def visit_Continue(self, node):
    self.continue_[-1].extend(self.head)
    self.head[:] = []

  def visit_Try(self, node):
    self.visit_statements(node.body)
    body = self.head
    handlers = []
    for handler in node.handlers:
      self.head = body[:]
      self.visit_statements(handler.body)
      handlers.extend(self.head)
    self.head = body
    self.visit_statements(node.orelse)
    self.head = handlers + self.head
    self.visit_statements(node.finalbody)


class Forward(object):
  """Forward analysis on CFG.

  Args:
    label: A name for this analysis e.g. 'active' for activity analysis. The
        AST nodes in the CFG will be given annotations 'name_in', 'name_out',
        'name_gen' and 'name_kill' which contain the incoming values, outgoing
        values, values generated by the statement, and values deleted by the
        statement respectively.
    gen: A function which takes the CFG node as well as a set of incoming
        values. It must return a set of newly generated values by the statement
        as well as a set of deleted (killed) values.
    op: Either the AND or OR operator. If the AND operator is used it turns
        into forward must analysis (i.e. a value will only be carried forward
        if it appears on all incoming paths). The OR operator means that
        forward may analysis is done (i.e. the union of incoming values will be
        taken).
  """

  def __init__(self, label, gen, op=operator.or_):
    self.gen = gen
    self.op = op
    self.out_label = label + '_out'
    self.in_label = label + '_in'
    self.gen_label = label + '_gen'
    self.kill_label = label + '_kill'

  def visit(self, node):
    if node.value:
      if anno.hasanno(node.value, self.out_label):
        before = hash(anno.getanno(node.value, self.out_label))
      else:
        before = None
      preds = [anno.getanno(pred.value, self.out_label)
               for pred in node.prev
               if anno.hasanno(pred.value, self.out_label)]
      if preds:
        incoming = functools.reduce(self.op, preds[1:], preds[0])
      else:
        incoming = frozenset()
      anno.setanno(node.value, self.in_label, incoming, safe=False)
      gen, kill = self.gen(node, incoming)
      anno.setanno(node.value, self.gen_label, gen, safe=False)
      anno.setanno(node.value, self.kill_label, kill, safe=False)
      anno.setanno(node.value, self.out_label, (incoming - kill) | gen,
                   safe=False)
      if hash(anno.getanno(node.value, self.out_label)) != before:
        for succ in node.next:
          self.visit(succ)
    else:
      preds = [anno.getanno(pred.value, self.out_label)
               for pred in node.prev]
      self.exit = functools.reduce(self.op, preds[1:], preds[0])


def forward(node, analysis):
  """Perform a given analysis on all functions within an AST."""
  if not isinstance(analysis, Forward):
    raise TypeError('not a valid forward analysis object')
  for succ in gast.walk(node):
    if isinstance(succ, gast.FunctionDef):
      #orig_args = succ.args.args[:]
      #for attr in get_self_attrs(node):
      #  succ.args.args.append(gast.Name(id=attr, ctx=None, annotation=None))
      cfg_obj = CFG.build_cfg(succ)
      analysis.visit(cfg_obj.entry)
      #succ.args.args = orig_args
  return node


class ReachingDefinitions(Forward):
  """Perform reaching definition analysis.

  Each statement is annotated with a set of (variable, definition) pairs.

  """

  def __init__(self):
    def definition(node, incoming):
      definitions = ast_.get_updated(node.value)
      gen = frozenset((id_, node.value) for id_ in definitions)
      kill = frozenset(def_ for def_ in incoming
                       if def_[0] in definitions)

      # prevent optimizer from killing assigns with subscript on LHS
      #if kill and isinstance(node.value, gast.Assign):
      #  k = []
      #  for def_ in kill:
      #    for t in node.value.targets:
      #      if isinstance(t, gast.Subscript) and def_[0] == ast_.get_name(t):
      #        break
      #    else:
      #      k.append(def_)
      #  kill = frozenset(k)


      if kill:
        import astunparse
        print('statement:', astunparse.unparse(node.value), end='')
        l = list(x[0] for x in incoming)
        for k, n in kill:
          print(k, ':', astunparse.unparse(n).strip())
      #   print('deinitions:', list(definitions))
      #   print('incoming:', l)
      #   print('kill:', list(x[0] for x in kill))

      return gen, kill
    super(ReachingDefinitions, self).__init__('definitions', definition)


class Defined(Forward):
  """Perform defined variable analysis.

  Each statement is annotated with a set of variables which are guaranteed to
  be defined at that point.
  """

  def __init__(self):
    def defined(node, incoming):
      gen = ast_.get_updated(node.value)
      return gen, frozenset()
    super(Defined, self).__init__('defined', defined, operator.and_)


class Active(Forward):
  """Active variable analysis.

  Given a set of active arguments, find all variables that are active i.e.
  variables whose values possibly depend on the given set of arguments.

  Args:
    wrt: A tuple of indices of arguments that are active.
  """

  def __init__(self, wrt):
    def active(node, incoming):
      gen = set()
      kill = set()
      if isinstance(node.value, gast.arguments):
        gen.update(node.value.args[i].id for i in wrt)
      if isinstance(node.value, gast.Assign):
        # Special-case e.g. x = tangent.pop(_stack)
        # such that all values popped off the stack are live.
        if anno.getanno(node.value.value, 'func', False) == utils.pop:
          gen.update(ast_.get_updated(node.value))
        else:
          names = get_variables(node.value.value)
          #for succ in gast.walk(node.value.value):
          #  if isinstance(succ, gast.Name) and succ.id in incoming:
          #    gen.update(ast_.get_updated(node.value))
          #    break
          #  elif isinstance(succ, gast.Attribute) and ast_.get_name(succ) in incoming:
          #    gen.update(ast_.get_updated(node.value))
          #    break
          #else:
          #  kill.update(ast_.get_updated(node.value))
          for n in names:
            if n in incoming:
              gen.update(ast_.get_updated(node.value))
              break
          else:
            kill.update(ast_.get_updated(node.value))
      return gen, kill
    super(Active, self).__init__('active', active)


class SelfVarVisitor(gast.NodeVisitor):
    """
    A visitor that collects all self.* attribute accesses.
    """

    def __init__(self):
        super(SelfVarVisitor, self).__init__()
        self.self_attrs = set()

    def visit_Attribute(self, node):  # (value, attr)
      try:
        name = ast_.get_name(node)
      except TypeError:
        pass
      else:
        if name.startswith('self.'):
            self.self_attrs.add(name)


def get_self_attrs(node):
    visitor = SelfVarVisitor()
    visitor.visit(node)
    return visitor.self_attrs


class VarVisitor(gast.NodeVisitor):
    """
    A visitor that collects all variable accesses.
    """

    def __init__(self):
        super(VarVisitor, self).__init__()
        self.variables = set()

    def visit_Name(self, node):  # (id)
      self.variables.add(node.id)

    def visit_Attribute(self, node):  # (value, attr)
      try:
        name = ast_.get_name(node)
      except TypeError:
        pass
      else:
        self.variables.add(name)


def get_variables(node):
    visitor = VarVisitor()
    visitor.visit(node)
    return visitor.variables
