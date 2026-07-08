## integration

- **legacy_susfs** (maybe wont be update)
```
curl -LSs "https://raw.githubusercontent.com/xxblebleblexx/MultiSU/refs/heads/legacy/kernel/setup.sh" | bash -s legacy_susfs
```
- **legacy**
```
curl -LSs "https://raw.githubusercontent.com/xxblebleblexx/MultiSU/refs/heads/legacy/kernel/setup.sh" | bash -s legacy
```

## Instruction for legacy or legacy susfs
- patch with [manual hook](https://github.com/xxblebleblexx/manual_hook_fix.git) or [kprobes hook](https://kernelsu-next.github.io/webpage/pages/how-to-integrate-for-non-gki.html#:~:text=CONFIG%5FKPROBES%3Dy%20CONFIG%5FKPROBE%5FEVENTS%3Dy%20CONFIG%5FKSU%5FKPROBE%5FHOOKS%3Dy%20CONFIG%5FKSU%3Dy)
## ksu suported
- MamboSU
- RKSU
- Wild KSU
- KernelSU-Next
