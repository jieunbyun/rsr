API Reference
=============

The public API exposes three groups of functions: the core workflow,
system-function builders, and the reference-state search helpers used
internally by :func:`~rsr.rsr.run_ref_extraction_by_mcs`.

Core workflow
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rsr.rsr.run_ref_extraction_by_mcs
   rsr.rsr.get_comp_cond_sys_prob
   rsr.rsr.get_comp_cond_sys_prob_multi

System-function builders
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rsr.igraph_sfun.make_igraph_sfun_global_conn
   rsr.igraph_sfun.make_igraph_sfun_conn
   rsr.igraph_sfun.eval_global_conn_igraph
   rsr.igraph_sfun.eval_1od_connectivity_igraph
   rsr.igraph_sfun.nx_to_igraph

Reference-state search
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rsr.rsr.minimise_upper_states_random
   rsr.rsr.minimise_lower_states_random
   rsr.rsr.update_refs
   rsr.rsr.sample_new_comp_st_to_test
   rsr.rsr.classify_samples_with_indices
