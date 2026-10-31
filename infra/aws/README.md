# AWS demo environment (not deployed)

Nothing in this directory has been run against AWS. Every step below creates or uses billed
resources and needs owner approval first (PRD 4.1: about $0–$25 a month, with every performance
run approved separately).

## What the template creates

| Resource | Purpose | Cost notes (verify current pricing) |
| --- | --- | --- |
| EC2 `t3.small` + 30 GB gp3 | API, worker, Postgres and Redis on one host (not HA) | The main cost; stop the instance when the demo is idle |
| ECR repository (immutable tags) | Release images by digest | Storage only; lifecycle keeps 20 images |
| S3 archive bucket | Parquet history; lifecycle 90 d → IA, 365 d → Glacier, expire at 730 d | Small at demo volumes |
| S3 model bucket | Promoted model artifacts | Small |
| IAM roles via GitHub OIDC | `release` pushes images; `production` deploys through SSM | Free |
| AWS Budget | Email at 80% actual and 100% forecast of the monthly budget | Free |

No SSH: administration goes through SSM. Port 8000 is closed until you set `AllowedCidr`.

## Steps (owner-run)

1. `aws cloudformation deploy --template-file infra/aws/template.yaml --stack-name fraud-demo --capabilities CAPABILITY_NAMED_IAM --parameter-overrides AlertEmail=<you> GitHubRepo=<owner/repo>`
2. Store secrets as SSM SecureString parameters under `/fraud/*`, then write `/opt/fraud/secrets.env`
   on the host (via SSM Session Manager). Copy `compose.prod.yaml` and `deploy.sh` to `/opt/fraud/`.
   Generate every value fresh for the cloud with `openssl rand -hex 32` (database passwords go into
   a URL, so avoid base64). Never reuse a value from local development, tests or old commits, and
   never commit one.
3. Set the GitHub repository variables named in `.github/workflows/*.yml`, plus protected
   environments `release`, `production` and `ml-training` with required reviewers.
4. Run the `release-image` workflow, then `deploy` with the image digest and the promoted model's
   manifest digest and S3 prefix.
5. When finished: stop or terminate the instance, and delete the stack to stop all charges. Keep
   the start, end and teardown times and the bill for the final report (T-506).

## Merchant identity in the cloud

The local dev issuer is test-only. For a cloud demo, configure a managed OIDC issuer (I-04) and
set `FRAUD_JWT_ISSUER`, `FRAUD_JWT_AUDIENCE` and `FRAUD_JWKS_URL`. The dashboard needs the same for
`FRAUD_OIDC_*`.
