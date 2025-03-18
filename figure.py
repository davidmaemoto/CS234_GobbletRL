import graphviz

dot = graphviz.Digraph(comment='MCTS Tree Visualization')
dot.attr(rankdir='LR')

dot.node('root', '', image='root.png', shape='none', width='0.8', height='0.8')

dot.node('a1', '', image='a1.png', shape='none', width='0.8', height='0.8')
dot.node('a2', '', image='an.png', shape='none', width='0.8', height='0.8')

dot.attr('edge', arrowsize='1.0', arrowlength='20.0')
dot.edge('root', 'a1', 'a₁')
dot.edge('root', 'a2', 'a₂')

dot.render('mcts_visualization', format='png', cleanup=True)