""" Running the model."""
from statistic import Statistic
from edmot import EdMot
from param_parser import parameter_parser
from utils import tab_printer, graph_reader,membership_saver
import time

def main(network_name,M,runs):
    """
    Parsing command line parameters, reading data, fitting EdMot and scoring the model.
    """
    graph1 = object()
    for no in range(runs):
        args = parameter_parser(no)
        tab_printer(args)
        graph = graph_reader(args.edge_path)
        graph1 = graph
        EdMot_start = time.process_time()
        model = EdMot(graph, args.components, args.cutoff, M)
        memberships = model.fit()
        EdMot_end = time.process_time()
        print("\nEdMot_{} spendtime={}s".format(no, EdMot_end - EdMot_start))
        membership_saver(args.membership_path, memberships)
    Statistic(graph1, network_name, M, runs).fit()

if __name__ == "__main__":
    runs = 1 #独立运行次数
    network_name = "PGP"
    M = "M1"
    main(network_name, M, runs)
