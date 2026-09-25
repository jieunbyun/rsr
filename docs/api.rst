API Reference
=============

The public API exposes four groups of functions: the core workflow,
system-function builders, the reference-state search helpers used
internally by :func:`~rsr.rsr.run_ref_extraction_by_mcs`, and the
active-acquisition rule that chooses which unknown sample it resolves next.

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

Active acquisition
------------------

Deciding which unknown sample to resolve next, used by
:func:`~rsr.rsr.run_ref_extraction_by_mcs` when ``active_ref_search`` is
enabled.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rsr.rsr.select_refs_by_acquisition
   rsr.rsr.compute_deficits
   rsr.rsr.acquisition_score
   rsr.rsr.refs_mat_to_states
   rsr.rsr.validate_ref_consistency
