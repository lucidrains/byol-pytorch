import Pyro4
import types

@Pyro4.expose
@Pyro4.callback
@Pyro4.oneway
def register_result(self, id=None, result=None):
    self.logger.debug('DISPATCHER: job %s finished'%(str(id)))
    with self.runner_cond:
        self.logger.debug('DISPATCHER: register_result: lock acquired')
        # fill in missing information
        job = self.running_jobs[id]
        job.time_it('finished')
        job.result = result['result']
        job.exception = result['exception']

        self.logger.debug('DISPATCHER: job %s on %s finished'%(str(job.id),job.worker_name))
        self.logger.debug(str(job))

        # delete job
        del self.running_jobs[id]

        # label worker as idle again
        try:
            # Shutdown worker after result is registered
            self.logger.info(f"Result for {job.id} registered, shutdown worker")
            self.worker_pool[job.worker_name].runs_job = None
            self.worker_pool[job.worker_name].proxy.shutdown()
            # notify the job_runner to check for more jobs to run
            self.runner_cond.notify()
        except KeyError:
            # happens for crashed workers, but we can just continue
            pass
        except:
            raise

    # call users callback function to register the result
    # needs to be with the condition released, as the master can call
    # submit_job quickly enough to cause a dead-lock
    self.new_result_callback(job)


def add_shutdown_worker_to_register_result(dispatcher):
    dispatcher.register_result = types.MethodType(register_result, dispatcher)