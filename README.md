# AGENTIC-AI
Agents that truly understands complex documents in the form of knowledge-graph and relies heavily on context.

## Deployed with cloudflare workers

- Uses cloudflare agent.
- Uses KV caching.

## What's inside?

This is a Turborepo includes the following packages/apps:

### Apps and Packages

- `docs`: a [Next.js](https://nextjs.org/) app
- `web`: another [Next.js](https://nextjs.org/) app
- `@repo/ui`: a stub React component library shared by both `web` and `docs` applications
- `@repo/eslint-config`: `eslint` configurations (includes `eslint-config-next` and `eslint-config-prettier`)
- `@repo/typescript-config`: `tsconfig.json`s used throughout the monorepo
-  `@packages/working-agent-python`: An agent built with langgraph, postgresql database to manage state, python.

### Here's the LangSmith tracing of the agent and the loop in which agent is working - 

<img width="1912" height="947" alt="Screenshot 2026-03-17 235123" src="https://github.com/user-attachments/assets/dc1cce78-a752-44dc-af94-bd9951797523" />

![Alt Text](packages/working-agent-python/graph.png)


### Build

To build all apps and packages, run the following command:

With [global `turbo`](https://turborepo.dev/docs/getting-started/installation#global-installation) installed (recommended):

```sh
cd my-turborepo
turbo build
```
