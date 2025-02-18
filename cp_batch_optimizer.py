from ortools.sat.python import cp_model
from qubots.base_optimizer import BaseOptimizer
from collections import defaultdict

class ConstraintProgrammingBatchOptimizer(BaseOptimizer):
    """
    Fixed CP optimizer with proper interval variable handling
    """
    
    def __init__(self, time_limit=300, num_search_workers=4):
        self.time_limit = time_limit
        self.num_search_workers = num_search_workers

    def optimize(self, problem, initial_solution=None, **kwargs):
        model = cp_model.CpModel()
        
        # Create variables and intervals
        task_intervals = self._create_task_intervals(model, problem)
        batch_assignments = self._create_batch_vars(model, problem)
        
        # Add constraints
        self._add_resource_constraints(model, problem, task_intervals)
        self._add_batch_constraints(model, problem, task_intervals, batch_assignments)
        self._add_precedence_constraints(model, problem, task_intervals)
        
        # Set objective
        makespan = model.NewIntVar(0, problem.time_horizon, 'makespan')
        model.AddMaxEquality(makespan, [entry['interval'].EndExpr() for entry in task_intervals])
        model.Minimize(makespan)

        
        # Solve and return
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_search_workers
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("Solution status is: ", status)
            return self._extract_solution(solver, problem, task_intervals)
        return problem.random_solution(), float('inf')

    def _create_task_intervals(self, model, problem):
        """Create interval variables for each task"""
        intervals = []
        for t in range(problem.nb_tasks):
            start = model.NewIntVar(0, problem.time_horizon, f'task{t}_start')
            end = model.NewIntVar(0, problem.time_horizon, f'task{t}_end')
            duration = problem.duration[t]
            interval = model.NewIntervalVar(start, duration, end, f'task{t}_interval')
            intervals.append({
                'start': start,
                'end': end,
                'interval': interval,
                'resource': problem.resources[t],
                'type': problem.types[t]
            })
        return intervals

    def _create_batch_vars(self, model, problem):
        """Create batch assignment variables"""
        return [model.NewIntVar(0, problem.nb_tasks, f'batch_{t}') 
                for t in range(problem.nb_tasks)]

    def _add_resource_constraints(self, model, problem, intervals):
        """Add resource non-overlap constraints"""
        resource_intervals = defaultdict(list)
        for interval in intervals:
            resource_intervals[interval['resource']].append(interval['interval'])
        
        for resource in resource_intervals:
            model.AddNoOverlap(resource_intervals[resource])

    def _add_batch_constraints(self, model, problem, intervals, batch_vars):
        """Add batch constraints (capacity and type homogeneity)"""
        batch_groups = defaultdict(list)
        for t, var in enumerate(batch_vars):
            key = (intervals[t]['resource'], intervals[t]['type'])
            batch_groups[key].append((t, var))
        
        for (resource, type_), tasks in batch_groups.items():
            capacity = problem.capacity[resource]
            # Instead of using a set of variables, iterate over possible batch indices.
            for b in range(problem.nb_tasks + 1):
                in_batch = []
                for t, var in tasks:
                    # Create a boolean variable that is True if task t is assigned batch b.
                    bool_var = model.NewBoolVar(f'in_batch_{b}_{t}')
                    model.Add(var == b).OnlyEnforceIf(bool_var)
                    model.Add(var != b).OnlyEnforceIf(bool_var.Not())
                    in_batch.append(bool_var)
                # Enforce that the number of tasks assigned to batch b is at most the capacity.
                model.Add(sum(in_batch) <= capacity)
            
            # (Optional) If additional type consistency constraints are needed,
            # make sure they're correctly formulated. The following line might be redundant
            # if each `var` is already the corresponding element in `batch_vars`.
            for t, var in tasks:
                model.Add(var == batch_vars[t])


    def _add_precedence_constraints(self, model, problem, intervals):
        """Add task precedence constraints"""
        for t in range(problem.nb_tasks):
            for s in problem.successors[t]:
                model.Add(intervals[t]['end'] <= intervals[s]['start'])

    def _extract_solution(self, solver, problem, intervals):
        """Convert solver output to qubots format"""
        batches = defaultdict(list)
        for t, interval in enumerate(intervals):
            resource = interval['resource']
            start = solver.Value(interval['start'])
            end = solver.Value(interval['end'])
            batches[resource].append((start, end, t))
        
        batch_schedule = []
        for resource in batches:
            # Sort batches by start time
            sorted_batches = sorted(batches[resource], key=lambda x: x[0])
            current_batch = []
            current_end = -1
            
            for start, end, task in sorted_batches:
                if start >= current_end:
                    # New batch: add the previous batch if exists
                    if current_batch:
                        batch_schedule.append({
                            'resource': resource,
                            'tasks': [task_tuple[2] for task_tuple in current_batch],
                            'start': current_batch[0][0],
                            'end': current_batch[-1][1]
                        })
                    current_batch = [(start, end, task)]
                    current_end = end
                else:
                    # Continue current batch by appending the entire tuple
                    current_batch.append((start, end, task))
                    current_end = max(current_end, end)
            
            # Add last batch if any
            if current_batch:
                batch_schedule.append({
                    'resource': resource,
                    'tasks': [task_tuple[2] for task_tuple in current_batch],
                    'start': current_batch[0][0],
                    'end': current_end
                })
        
        return {'batch_schedule': batch_schedule}, solver.ObjectiveValue()
