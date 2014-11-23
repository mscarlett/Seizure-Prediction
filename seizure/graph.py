'''
Created on Oct 18, 2014

@author: newuser
'''
import igraph

from seizure.transform import _IdentityTransform

class DependencyGraph(object):
    """
    Create a pipeline for applying transformations to input data by creating a
    dependency graph. The dependency graph is defined as the digraph with
    unique vertices representing the transforms and edges that equate the
    output of one transform with the input of another.
    """
    def __init__(self, transforms, name):
        self.name = name
        self.build_dependency_graph(transforms)
        
    def build_dependency_graph(self, transforms):
        """
        Constructs the dependency graph from the final transforms.
        """
        try:
            self.final_transforms = tuple(transforms)
        except TypeError:
            self.final_transforms = tuple([transforms])
        self.graph = igraph.Graph(directed=True)
        self.base_transforms = []
        # Recursively adds dependencies required to generate each feature
        for transform in self.final_transforms:
            self.__build_dependency_graph(transform)
        # The identity transform is where raw data is input initially
        # We add a self-loop so that it outputs its own input
        identity = _IdentityTransform()
        self.graph.add_vertex(label=identity.get_name(), transform=identity, data=None, color="yellow")
        self.root = self.get(identity)
        # Connects the identity transform to those that process raw data
        for base_transform in self.base_transforms:
            self.graph.add_edge(self.root, self.get(base_transform))
                   
    def __build_dependency_graph(self, transform):
        """
        Adds the current vertex and its dependencies to the dependency graph.
        """
        if self.get(transform) == None:
            color = "orange" if transform in self.final_transforms else "red"
            self.graph.add_vertex(label=transform.get_name(), transform=transform, color=color)
            for dependency in transform.requires():
                self.__build_dependency_graph(dependency)
                self.graph.add_edge(self.get(dependency), self.get(transform))
            if len(transform.requires()) == 0:
                self.base_transforms.append(transform)
        
    def get_name(self):
        return self.name
    
    def apply(self, data):
        """
        Transforms the input data into the feature set.
        """
        results = dict()
        # Traverse the dependency graph
        self.traverse(data, results)
        # Obtain the output of the graph
        return self.get_output(results)
    
    def traverse(self, data, results):
        # Feed the input data into the root vertex
        results[self.root] = data
        visited = {self.root}
        # Traverse dependency tree starting from base nodes to obtain output
        for base_transform in self.base_transforms:
            self.__dependency_traverse(self.get(base_transform), visited, results)
        
    def get_output(self):
        raise NotImplementedError("You forgot to override this method.")
    
    def __dependency_traverse(self, v, visited, results):
        """
        Recursively traverse the dependency graph.
        """
        if v in visited:
            return
        visited.add(v)
        # Create list of input data
        data = []
        for p in v.predecessors():
            # Traverse dependencies that have not been visited yet
            self.__dependency_traverse(p, visited, results)
            data.append(results[p])
        data = data[0] if len(data) == 1 else data
        # Flatten list of input data if there is only one element
        results[v] = self.traverse_vertex(v, data)
        # Traverse transforms that require this dependency and have not been visited
        for n in v.successors():
            self.__dependency_traverse(n, visited, results)
    
    def traverse_vertex(self, vertex, data):
        return vertex["transform"].apply(data)
    
    def get(self, transform):
        """
        Retrieve the vertex from the graph with the same transform.
        """
        for v in self.graph.vs:
            if v["transform"] == transform:
                return v
            
    def erase(self):
        for v in self.graph.vs:
            v["data"] = None
    
    def plot(self):
        """
        Plot the dependency graph.
        """
        layout = self.graph.layout("kk")
        bbox = igraph.BoundingBox(600, 600)
        figure = igraph.Plot(bbox=bbox, background="white")
        bbox = bbox.contract(100)
        figure.add(self.graph, layout = layout, bbox=bbox)
        figure.show()