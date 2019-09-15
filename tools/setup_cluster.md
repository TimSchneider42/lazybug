## How to setup the cluster:
1. Generate ssh-key using `ssh-keygen`, if you already have an ssh-key you can use that as well of course
2. Upload ssh-key to cluster using `ssh-copy-id -i <path/to/key> <tu-id>@lcluster7.hrz.tu-darmstadt.de`
3. Open ssh-config : `nano ~/.ssh/config`
4. Add \
`Host hrz7` \
`User <tu-id>` \
`HostName lcluster7.hrz.tu-darmstadt.de`
5. Run script `./sync` in folder `tools`
6. Now log in to the cluster `ssh hrz7`
7. Run `./setup` in folder `tools/cluster` on the cluster and select yes every time
8. Now everything should be setup correctly

Jobs can be started via `submit <job> <args>` where job is any of `training`, `import_bpgn`, `shuffle_db` or `preprocessing`.
