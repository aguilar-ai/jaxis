import { Footer, Layout, Navbar } from "nextra-theme-docs";
import { Banner, Head, Image } from 'nextra/components'
import { inter, interTight, jersey25, googleSansCode } from "@/lib/fonts";
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import jaxisLogo from "@/public/logo-red.svg";
import "@/app/globals.css";
import 'katex/dist/katex.min.css'


export const metadata = {
  title: "Jaxis",
  description: "Jaxis is validating axis semantics of machine learning training pipelines using property testing.",
}

const navbar = (
  <Navbar
    logo={
      <Image src={jaxisLogo} alt="Jaxis" width={96} height={96} />
    }
    projectLink="https://github.com/aguilar-ai/jaxis"
  />
)

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      dir="ltr"
      suppressHydrationWarning
    >
      <Head
      // TODO: Add additional tags
      />
      <body className={`${inter.variable} ${interTight.variable} ${jersey25.variable} ${googleSansCode.variable}`}>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/shuding/nextra/tree/main/docs"
          sidebar={{ autoCollapse: true }}
          darkMode={false}
          nextThemes={{
            defaultTheme: "light",
          }}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}