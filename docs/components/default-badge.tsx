import { Badge } from "./ui/badge";

const DefaultBadge = ({ value }: { value: string | number }) => {
  return (
    <Badge
      variant="outline"
      className="mt-6 font-mono uppercase text-muted-foreground"
    >
      Default{" "}
      <span className="font-extrabold text-primary">
        {typeof value === "number" && value < 1 && value > 0
          ? value.toExponential()
          : value}
      </span>
    </Badge>
  );
};

export default DefaultBadge;
