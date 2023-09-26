import sys

import pgmpy.base.DAG
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

import util

samples = util.get_prepared_base_samples()

# 1. Learning Structure

scoring_method = K2Score(data=samples)  # BDeuScore | AICScore
estimator = HillClimbSearch(data=samples)

dag: pgmpy.base.DAG = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4, epsilon=30,
)

regular = '#a1b2ff'  # blue
special = '#c46262'  # red

util.print_BN(dag, vis_ls=["circo"], save=True, name="raw_model", color_map=regular)

# Removing wrong edges
# dag.remove_edge("pixel", "GPU")  # Simply wrong
# dag.remove_edge("bitrate", "config")  # Simply wrong
# dag.remove_edge("transformed", "GPU")  # Simply wrong
# dag.remove_edge("transformed", "delay")  # Correlated but not causal
# dag.remove_edge("delay", "consumption")  # Correlated but not causal
# dag.remove_edge("delay", "CPU")  # Correlated but not causal
# dag.remove_edge("config", "memory")  # This is rather the device type
# dag.remove_edge("GPU", "config")  # This is rather the device type, but this was removed...
#
# # Reversing edges
# dag = util.fix_edge_between_u_v(dag, "GPU", "delay")
# dag = util.fix_edge_between_u_v(dag, "fps", "bitrate")
# dag = util.fix_edge_between_u_v(dag, "config", "consumption")
# dag = util.fix_edge_between_u_v(dag, "config", "delay")
#
# # Bitrate correction
# dag.remove_edge("bitrate", "transformed")
# dag.add_edge("pixel", "transformed")
# dag.remove_edge("bitrate", "distance")
# dag.add_edge("fps", "distance")
# dag.remove_edge("bitrate", "delay")
# dag.add_edge("pixel", "delay")

# util.print_BN(dag, vis_ls=["circo"], save=True, name='refined_model', color_map=regular)  # dot!
# dag.remove_edge("transformed", "delay")  # Correlated but not causal


# util.print_BN(util.get_mb_as_bn(model=dag, center="bitrate"), root="bitrate", save=True, vis_ls=["dot"],
#          color_map=[regular, regular, regular, regular, regular, regular, regular, special])
# util.print_BN(util.get_mb_as_bn(model=dag, center="distance"), root="distance", save=True,
#               color_map=[regular, regular, special])
# util.print_BN(util.get_mb_as_bn(model=dag, center="transformed"), root="transformed", save=True,
#               color_map=[regular, regular, regular, special, regular, regular, regular])
# util.print_BN(util.get_mb_as_bn(model=dag, center="consumption"), root="consumption", save=True,
#               color_map=[special, regular, regular, regular])
# util.print_BN(util.get_mb_as_bn(model=dag, center="delay"), root="delay", save=True,
#          color_map=[special, regular, regular, regular, regular])
# util.print_BN(util.get_mb_as_bn(model=dag, center="fps"), vis_ls=["circo"], root="fps", save=True,
#          color_map=[regular, special, regular, regular, regular])

print("Structure Learning Finished")

# 2. Learning Parameters

model = BayesianNetwork(ebunch=dag.edges())
model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

print("Parameter Learning Finished")

XMLBIFWriter(model).write_xmlbif('model.xml')
print("Model exported as 'model.xml'")

sys.exit()
